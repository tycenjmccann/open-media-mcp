"""open-media-mcp GPU worker: pulls jobs from SQS, renders them in the local ComfyUI, ships results to S3.

One job at a time per GPU. Durable by design:
  * the SQS lease is extended every minute while a job renders; if the box dies (spot reclaim) the message
    reappears and another worker retries it (3 attempts, then the dead-letter queue);
  * a spot interruption notice hands the job back to the queue right away;
  * cancel_job flips the DynamoDB status; the heartbeat sees it and interrupts ComfyUI.
After MEDIA_IDLE_MINUTES with an empty queue the worker removes its own instance from the Auto Scaling group.
"""
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import av
import boto3
import imageio_ffmpeg
from botocore.config import Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as backend_registry  # noqa: E402
import identity  # noqa: E402  (shared/identity.py, deployed next to this file)
import prompting  # noqa: E402

VERSION = "0.1.0"
E = os.environ
REGION = E.get("AWS_DEFAULT_REGION", "us-east-1")
COMFY = "http://127.0.0.1:8188"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}
MAX_INPUT_BYTES = 500 * 1024 * 1024
LEASE_SECONDS = 900

cfg = Config(region_name=REGION, retries={"max_attempts": 8, "mode": "adaptive"})
sqs = boto3.client("sqs", config=cfg)
s3 = boto3.client("s3", config=cfg)
table = boto3.resource("dynamodb", config=cfg).Table(E["MEDIA_TABLE"])
rek = boto3.client("rekognition", config=cfg)
events = boto3.client("events", config=cfg)
cw = boto3.client("cloudwatch", config=cfg)
cwlogs = boto3.client("logs", config=cfg)
asg = boto3.client("autoscaling", config=cfg)
ec2 = boto3.client("ec2", config=cfg)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("worker")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def imds(path):
    tok = urllib.request.urlopen(urllib.request.Request(
        "http://169.254.169.254/latest/api/token", method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}), timeout=2).read().decode()
    req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}",
                                 headers={"X-aws-ec2-metadata-token": tok})
    try:
        return urllib.request.urlopen(req, timeout=2).read().decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


INSTANCE = {"instance_id": imds("instance-id"), "instance_type": imds("instance-type"),
            "lifecycle": imds("instance-life-cycle"), "az": imds("placement/availability-zone")}


class CloudLog:
    """Job-level events to CloudWatch Logs (one stream per instance); stdout keeps the chatty detail."""

    def __init__(self, group):
        self.group, self.stream = group, INSTANCE["instance_id"]
        try:
            cwlogs.create_log_stream(logGroupName=group, logStreamName=self.stream)
        except cwlogs.exceptions.ResourceAlreadyExistsException:
            pass

    def __call__(self, **fields):
        fields.setdefault("ts", now())
        line = json.dumps(fields, default=str)
        log.info(line)
        try:
            cwlogs.put_log_events(logGroupName=self.group, logStreamName=self.stream,
                                  logEvents=[{"timestamp": int(time.time() * 1000), "message": line}])
        except Exception as exc:  # logging must never take a render down
            log.warning("cloudwatch log failed: %s", exc)


audit = CloudLog(E["MEDIA_LOG_GROUP"])


def metric(name, value=1.0, unit="Count"):
    try:
        cw.put_metric_data(Namespace=E["MEDIA_METRIC_NAMESPACE"], MetricData=[{
            "MetricName": name, "Value": float(value), "Unit": unit,
            "Dimensions": [{"Name": "Stack", "Value": E["MEDIA_STACK"]}]}])
    except Exception as exc:
        log.warning("metric %s failed: %s", name, exc)


_PRICE = {}


def usd_per_hour():
    """Live price of this instance: the AZ's current spot price, or the on-demand list price."""
    if "v" in _PRICE and time.time() - _PRICE["t"] < 900:
        return _PRICE["v"], _PRICE["basis"]
    price, basis = None, ""
    try:
        if INSTANCE["lifecycle"] == "spot":
            h = ec2.describe_spot_price_history(InstanceTypes=[INSTANCE["instance_type"]], AvailabilityZone=INSTANCE["az"],
                                                ProductDescriptions=["Linux/UNIX"], MaxResults=1)["SpotPriceHistory"]
            price, basis = float(h[0]["SpotPrice"]), f"spot price, {INSTANCE['az']}"
        else:
            pricing = boto3.client("pricing", region_name="us-east-1")
            loc = {"us-east-1": "US East (N. Virginia)", "us-east-2": "US East (Ohio)", "us-west-2": "US West (Oregon)"}.get(REGION)
            items = pricing.get_products(ServiceCode="AmazonEC2", MaxResults=1, Filters=[
                {"Type": "TERM_MATCH", "Field": f, "Value": v} for f, v in (
                    ("instanceType", INSTANCE["instance_type"]), ("location", loc), ("operatingSystem", "Linux"),
                    ("tenancy", "Shared"), ("preInstalledSw", "NA"), ("capacitystatus", "Used"))])["PriceList"]
            od = json.loads(items[0])["terms"]["OnDemand"]
            price = float(next(iter(next(iter(od.values()))["priceDimensions"].values()))["pricePerUnit"]["USD"])
            basis = "on-demand list price"
    except Exception as exc:
        log.warning("price lookup failed: %s", exc)
    _PRICE.update(v=price, basis=basis, t=time.time())
    return price, basis


def job_cost(gpu_seconds):
    price, basis = usd_per_hour()
    return {"gpu_seconds": round(gpu_seconds, 1), "usd_per_hour": price, "basis": basis,
            "usd": round(price * gpu_seconds / 3600, 5) if price else None,
            "instance_type": INSTANCE["instance_type"], "lifecycle": INSTANCE["lifecycle"],
            "scope": "this job's time on the GPU; boot and idle time are fleet overhead (see cost_report)"}


def status_file(phase, job_id=None):
    body = {**INSTANCE, "phase": phase, "job_id": job_id, "updated": now(), "version": VERSION}
    s3.put_object(Bucket=E["MEDIA_BUCKET"], Key=f"status/workers/{INSTANCE['instance_id']}.json",
                  Body=json.dumps(body).encode(), ContentType="application/json")


def ddb_safe(v):
    """DynamoDB wants Decimal, not float."""
    return json.loads(json.dumps(v, default=str), parse_float=Decimal)


def update_job(job_id, cond_not_final=True, **fields):
    names = {f"#{k}": k for k in fields}
    values = {f":{k}": ddb_safe(v) for k, v in fields.items()}
    expr = "SET " + ", ".join(f"#{k} = :{k}" for k in fields)
    kw = {}
    if cond_not_final:
        kw["ConditionExpression"] = "NOT (#status IN (:s1, :s2, :s3, :s4))"
        names["#status"] = "status"
        values.update({":s1": "SUCCEEDED", ":s2": "FAILED", ":s3": "BLOCKED", ":s4": "CANCELLED"})
    try:
        table.update_item(Key={"job_id": job_id}, UpdateExpression=expr, ExpressionAttributeNames=names,
                          ExpressionAttributeValues=values, **kw)
        return True
    except table.meta.client.exceptions.ConditionalCheckFailedException:
        return False


def emit(job, status, extra=None):
    detail = {"job_id": job["job_id"], "status": status, "tenant": job.get("tenant"),
              "backend": json.loads(job["spec"])["backend"], "labels": job.get("labels", {}), **(extra or {})}
    try:
        events.put_events(Entries=[{"Source": "open-media-mcp", "DetailType": "Media Job Status Changed",
                                    "Detail": json.dumps(detail, default=str)}])
    except Exception as exc:
        log.warning("event failed: %s", exc)


class JobError(Exception):
    """Not retryable: bad input or content policy."""

    def __init__(self, msg, status="FAILED", details=None):
        super().__init__(msg)
        self.status, self.details = status, details or {}


class Cancelled(Exception):
    pass


class Preempted(Exception):
    pass


# --------------------------------------------------------------------------- ComfyUI
def comfy_get(path, timeout=120):
    with urllib.request.urlopen(COMFY + path, timeout=timeout) as r:
        return r.read()


def comfy_post(path, body):
    req = urllib.request.Request(COMFY + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise JobError(f"ComfyUI rejected the graph: {e.read().decode()[:2000]}")


def comfy_upload(path):
    # /upload/image stores any file type in input/ (the ComfyUI frontend uses it for video and audio too).
    boundary = uuid.uuid4().hex
    name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(path)}"
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        data = fh.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(COMFY + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["name"]


def wait_for_comfy():
    for _ in range(360):
        try:
            comfy_get("/system_stats", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise SystemExit("ComfyUI never came up")


# --------------------------------------------------------------------------- media
def ffmpeg(*args):
    r = subprocess.run([FFMPEG, "-v", "error", "-y", *args], capture_output=True, text=True)
    if r.returncode:
        raise JobError(f"ffmpeg failed: {r.stderr[-800:]}")


def probe(path):
    try:
        with av.open(path) as c:
            dur = float(c.duration / av.time_base) if c.duration else 0.0
            kinds = {s.type for s in c.streams}
    except Exception as exc:
        raise JobError(f"unreadable media {os.path.basename(path)}: {exc}")
    return dur, kinds


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


LOUDNESS = "I=-14:TP=-1.5:LRA=11"  # streaming-platform loudness target (EBU R128 measurement)
X264_HQ = ["-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=tv,format=yuv420p,"
                  "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def loudnorm_filter(path):
    """Two-pass loudnorm: measure first so the second pass can apply one linear gain instead of riding the level."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", path, "-vn",
                        "-af", f"loudnorm={LOUDNESS}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    try:
        m = json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
        if float(m["input_i"]) < -60:  # near-silent: leave it alone
            return None
    except (ValueError, KeyError):
        return None
    return (f"loudnorm={LOUDNESS}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
            f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")


def finish_video(raw, tmp, tuning, soundtrack=None, at=0.0):
    """One ffmpeg pass: mux a soundtrack (or master the model's own audio) and do the delivery encode.
    -> (path, record of what was applied). Recipe r0 = stream copy, so its output is byte-for-byte what ComfyUI wrote."""
    hq = tuning.get("encode") == "hq"
    _, kinds = probe(raw)
    rec = {"encode": "x264 crf16 slow bt709" if hq else "copy", "audio": None}
    args, fc, amap = ["-i", raw], None, None
    if soundtrack:
        args += ["-i", soundtrack]
        fc, amap, rec["audio"] = f"[1:a]adelay={int(round(at * 1000))}:all=1,apad[a]", "[a]", "soundtrack muxed"
    elif "audio" in kinds:
        amap, rec["audio"] = "0:a", "model audio"
        ln = loudnorm_filter(raw) if tuning.get("audio_master") else None
        if ln:
            fc, amap, rec["audio"] = f"[0:a]{ln},aresample=48000[a]", "[a]", f"model audio, loudnorm {LOUDNESS}, 48 kHz"
    if not hq and not fc and not soundtrack:
        return raw, rec
    out = os.path.join(tmp, "finished.mp4")
    audio = (["-map", amap, "-c:a", "aac", "-b:a", "192k"] + (["-ar", "48000"] if hq else [])) if amap else ["-an"]
    ffmpeg(*args, *(["-filter_complex", fc] if fc else []), "-map", "0:v", *(X264_HQ if hq else ["-c:v", "copy"]),
           *audio, *(["-shortest"] if soundtrack else []), "-movflags", "+faststart", out)
    return out, rec


def pad_audio(path, length_s, frame_s, prep_tool):
    """Pad with silence to a whole number of audio-VAE frames so the encoder doesn't crop the head of the guide."""
    target = (int(length_s / frame_s) + 1) * frame_s
    out = prep_tool.out(path, ".wav")
    ffmpeg("-i", path, "-af", f"apad=whole_dur={target:.3f}", "-ac", "2", "-ar", "48000", out)
    return out


class Prep:
    """Download each input once, trim/convert it to what the model expects, record hashes for provenance."""

    def __init__(self, tmp, fps, max_seconds):
        self.tmp, self.fps, self.max_seconds = tmp, fps, max_seconds
        self.inputs = []

    def fetch(self, key, role):
        head = s3.head_object(Bucket=E["MEDIA_BUCKET"], Key=key)
        if head["ContentLength"] > MAX_INPUT_BYTES:
            raise JobError(f"{key} is over {MAX_INPUT_BYTES >> 20} MB")
        local = os.path.join(self.tmp, f"{len(self.inputs):02d}_{os.path.basename(key)}")
        s3.download_file(E["MEDIA_BUCKET"], key, local)
        self.inputs.append({"role": role, "key": key, "bytes": head["ContentLength"], "sha256": sha256(local)})
        return local

    def out(self, src, ext):
        return os.path.join(self.tmp, f"{uuid.uuid4().hex[:6]}_{os.path.splitext(os.path.basename(src))[0]}{ext}")

    def image(self, key, role):
        path = self.fetch(key, role)
        if probe(path)[1] - {"video"}:
            raise JobError(f"{key} is not an image")
        return path

    def video(self, item, role, min_s=0.0):
        path = self.fetch(item["key"], role)
        dur, kinds = probe(path)
        if "video" not in kinds:
            raise JobError(f"{item['key']} has no video stream")
        t0 = float(item.get("from") or 0.0)
        t1 = min(float(item["to"]) if item.get("to") is not None else dur, dur, t0 + self.max_seconds)
        if t1 - t0 < min_s:
            raise JobError(f"{item['key']}: needs at least {min_s}s, got {t1 - t0:.1f}s")
        out = self.out(path, ".mp4")
        vf = f"fps={self.fps},scale='if(lt(iw,ih),min(768,iw),-2)':'if(lt(iw,ih),-2,min(768,ih))'"
        audio = ["-c:a", "aac", "-b:a", "192k"] if "audio" in kinds else ["-an"]
        ffmpeg("-ss", str(t0), "-to", str(t1), "-i", path, "-vf", vf, "-c:v", "libx264", "-crf", "16",
               "-pix_fmt", "yuv420p", *audio, out)
        return out, "audio" in kinds

    def audio(self, item, role):
        path = self.fetch(item["key"], role)
        dur, kinds = probe(path)
        if "audio" not in kinds:
            raise JobError(f"{item['key']} has no audio stream")
        t0 = float(item.get("from") or 0.0)
        t1 = min(float(item["to"]) if item.get("to") is not None else dur, dur, t0 + self.max_seconds)
        out = self.out(path, ".wav")
        ffmpeg("-ss", str(t0), "-to", str(t1), "-i", path, "-vn", "-ac", "2", "-ar", "48000", out)
        return out, t1 - t0


# --------------------------------------------------------------------------- moderation
BLOCKED = {s.strip() for s in E.get("MEDIA_BLOCKED_LABELS", "").split(",") if s.strip()}
MIN_CONF = float(E.get("MEDIA_MODERATION_MIN_CONFIDENCE", "80"))
MODERATE = E.get("MEDIA_MODERATION", "false").lower() == "true"


def moderate_jpegs(paths):
    """Amazon Rekognition on each frame; returns the blocked labels found (empty = clean)."""
    hits = []
    for p in paths:
        with open(p, "rb") as fh:
            labels = rek.detect_moderation_labels(Image={"Bytes": fh.read()}, MinConfidence=MIN_CONF)["ModerationLabels"]
        for lab in labels:
            if lab["Name"] in BLOCKED or lab.get("ParentName") in BLOCKED:
                hits.append({"frame": os.path.basename(p), "label": lab["Name"], "parent": lab.get("ParentName"),
                             "confidence": round(lab["Confidence"], 1)})
    return hits


def to_jpeg(src, out, at=None):
    args = (["-ss", f"{at:.3f}"] if at is not None else []) + ["-i", src, "-frames:v", "1",
                                                                "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", out]
    ffmpeg(*args)
    return out


def sample_frames(video, tmp, n=6):
    dur = probe(video)[0]
    return [to_jpeg(video, os.path.join(tmp, f"mod_{i}.jpg"), dur * (i + 0.5) / n) for i in range(n)]


# --------------------------------------------------------------------------- identity
def check_identity_video(cfg, video, tmp, n=8):
    """Rekognition face match of n evenly spaced frames against the reference images; never fails the job."""
    thr = {k: cfg[k] for k in ("pass_at", "review_at", "min_face") if k in cfg}
    try:
        datas = [s3.get_object(Bucket=E["MEDIA_BUCKET"], Key=k)["Body"].read() for k in cfg["reference"]]
        refs, counts = identity.reference_faces(rek, datas)
        if not refs:
            return {"verdict": "REVIEW", "reason": "no face found in the reference image(s)"}
        dur = probe(video)[0]
        frames = []
        for i in range(n):
            t = dur * (i + 0.5) / n
            with open(to_jpeg(video, os.path.join(tmp, f"id_{i}.jpg"), t), "rb") as fh:
                r = identity.score_image(rek, refs, fh.read(), **thr)
            r.pop("face_box", None)
            if r.get("matched_reference") is not None:
                r["matched_reference"] = cfg["reference"][r["matched_reference"]]
            frames.append({"t": round(t, 2), **r})
        return {**identity.aggregate_frames(frames, thr.get("pass_at", identity.PASS_AT),
                                            thr.get("review_at", identity.REVIEW_AT)),
                "frames": frames, "reference_faces": dict(zip(cfg["reference"], counts)),
                "thresholds": {"pass_at": thr.get("pass_at", identity.PASS_AT),
                               "review_at": thr.get("review_at", identity.REVIEW_AT),
                               "min_face": thr.get("min_face", identity.MIN_FACE)}}
    except Exception as exc:  # the render is still good; report the check as not done
        log.warning("identity check: %s", exc)
        return {"verdict": "ERROR", "reason": f"identity check failed: {str(exc)[:300]}"}


# --------------------------------------------------------------------------- heartbeat
class Heartbeat(threading.Thread):
    def __init__(self, receipt, job_id):
        super().__init__(daemon=True)
        self.receipt, self.job_id = receipt, job_id
        self.stop_evt = threading.Event()
        self.cancelled = self.preempted = False

    def run(self):
        while not self.stop_evt.wait(30):
            try:
                if imds("spot/instance-action"):
                    self.preempted = True
                    comfy_post("/interrupt", {})
                    return
                item = table.get_item(Key={"job_id": self.job_id}, ProjectionExpression="#s",
                                      ExpressionAttributeNames={"#s": "status"}).get("Item", {})
                if item.get("status") == "CANCELLED":
                    self.cancelled = True
                    comfy_post("/interrupt", {})
                    return
                sqs.change_message_visibility(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=self.receipt,
                                              VisibilityTimeout=LEASE_SECONDS)
            except Exception as exc:
                log.warning("heartbeat: %s", exc)

    def check(self):
        if self.preempted:
            raise Preempted()
        if self.cancelled:
            raise Cancelled()


# --------------------------------------------------------------------------- job
def run_job(job, hb, tmp):
    spec = json.loads(job["spec"])
    backend = BACKENDS.get(spec["backend"])
    if backend is None:
        raise RuntimeError(f"backend {spec['backend']} is not enabled on this worker (retried; a newer worker may have it)")
    job_id = job["job_id"]
    t_start = time.time()
    tuning = spec.get("tuning") or {}

    prep_tool = Prep(tmp, backend.fps or 24, 15.0)
    prep = {"refs": [], "ref_videos": [], "ref_audios": [], "guides": [], "soundtrack": None}
    for r in spec.get("refs", []):
        prep["refs"].append(prep_tool.image(r["key"], "ref"))
    for r in spec.get("ref_videos", []):
        prep["ref_videos"].append(prep_tool.video(r, "ref_video", min_s=2.0))
    for r in spec.get("ref_audios", []):
        prep["ref_audios"].append(prep_tool.audio(r, "ref_audio")[0])
    anchors = ([(spec["first_frame"], 0.0)] if spec.get("first_frame") else []) + \
              [(k["key"], float(k["at"])) for k in spec.get("keyframes", [])] + \
              ([(spec["last_frame"], "end")] if spec.get("last_frame") else [])
    for key, at in anchors:
        if key.lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".m4v")):
            prep["guides"].append(("clip", prep_tool.video({"key": key}, "anchor_clip")[0], at))
        else:
            prep["guides"].append(("image", prep_tool.image(key, "keyframe"), at))
    if spec.get("soundtrack"):
        prep["soundtrack"] = prep_tool.audio(spec["soundtrack"], "soundtrack")
        frame_s = getattr(backend, "audio_frame_s", None)
        if tuning.get("pad_audio") and frame_s:
            prep["soundtrack"] = (pad_audio(prep["soundtrack"][0], prep["soundtrack"][1], frame_s, prep_tool),
                                  prep["soundtrack"][1])
    hb.check()

    prompt_info = None
    if tuning.get("enhance_prompt"):
        update_job(job_id, phase="rewriting the prompt")
        spec["prompt"], prompt_info = prompting.enhance(spec, prep, backend.name, bool(tuning.get("keyframe_refs")))
        if "error" not in prompt_info:
            update_job(job_id, prompt_enhanced=spec["prompt"][:6000])
        if spec.get("aspect") == "auto" and prompt_info.get("aspect"):
            spec["aspect"] = prompt_info["aspect"]  # the expander picked the canvas (h3-image)
        hb.check()

    if MODERATE:
        stills = [p for p in prep["refs"]] + [p for kind, p, _ in prep["guides"] if kind == "image"]
        jpgs = [to_jpeg(p, os.path.join(tmp, f"in_{i}.jpg")) for i, p in enumerate(stills)]
        hits = moderate_jpegs(jpgs)
        if hits:
            raise JobError("an input image failed the content check", "BLOCKED", {"moderation": {"inputs": hits}})

    local = prep["refs"] + [p for p, _ in prep["ref_videos"]] + prep["ref_audios"] + \
        [p for _, p, _ in prep["guides"]] + ([prep["soundtrack"][0]] if prep["soundtrack"] else [])
    names = {p: comfy_upload(p) for p in dict.fromkeys(local)}
    graph = backend.build_graph(spec, prep, names, job_id)

    update_job(job_id, status="RUNNING", phase="rendering on GPU", render_started_at=now())
    status_file("rendering", job_id)
    t_render = time.time()
    pid = comfy_post("/prompt", {"prompt": graph, "client_id": job_id})["prompt_id"]
    hist = None
    while True:
        try:
            hb.check()
        except (Cancelled, Preempted):
            comfy_post("/queue", {"delete": [pid]})
            raise
        hist = json.loads(comfy_get(f"/history/{pid}")).get(pid)
        if hist:
            st = hist.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
                hb.check()
                raise RuntimeError(f"render failed: {json.dumps(msgs)[:2000]}")
            if st.get("completed"):
                break
        time.sleep(3)
    render_s = time.time() - t_render
    if getattr(backend, "kind", "video") == "audio":
        return finish_audio(job, spec, backend, hist, tmp, render_s, t_start, prep_tool.inputs, prompt_info)
    if getattr(backend, "kind", "video") == "image":
        return finish_image(job, spec, backend, hist, tmp, render_s, t_start, prep_tool.inputs, prompt_info)

    files = [f for out in hist["outputs"].values() for v in out.values() if isinstance(v, list)
             for f in v if isinstance(f, dict) and str(f.get("filename", "")).endswith((".mp4", ".webm", ".mov"))]
    if not files:
        raise RuntimeError(f"no video in outputs: {json.dumps(hist['outputs'])[:800]}")
    f = files[0]
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
    raw = os.path.join(tmp, "raw" + os.path.splitext(f["filename"])[1])
    with open(raw, "wb") as fh:
        fh.write(comfy_get(f"/view?{q}", timeout=1800))

    update_job(job_id, phase="checking and packaging")
    st = spec.get("soundtrack") or {}
    muxed = bool(prep["soundtrack"] and (st.get("keep_original", True) or not backend.native_audio))
    raw, finishing = finish_video(raw, tmp, tuning, prep["soundtrack"][0] if muxed else None,
                                  float(st.get("at") or 0.0))

    moderation = {"enabled": MODERATE}
    if MODERATE:
        hits = moderate_jpegs(sample_frames(raw, tmp))
        moderation.update(frames_checked=6, hits=hits)
        if hits:
            qkey = f"quarantine/{job_id}/{job_id}.mp4"
            s3.upload_file(raw, E["MEDIA_BUCKET"], qkey)
            raise JobError("the render failed the output content check (kept in quarantine for review)", "BLOCKED",
                           {"moderation": moderation, "quarantine_key": qkey})

    ident = None
    if spec.get("identity_check"):
        update_job(job_id, phase="checking the character")
        ident = check_identity_video(spec["identity_check"], raw, tmp)

    final = os.path.join(tmp, f"{job_id}.mp4")
    tags = {"comment": f"AI-generated by open-media-mcp {VERSION} ({backend.name}); job {job_id}",
            "ai_generated": "true", "generator": f"open-media-mcp/{VERSION}", "backend": backend.name,
            "job_id": job_id}
    if backend.notice:
        tags["copyright"] = backend.notice
    meta = [a for k, v in tags.items() for a in ("-metadata", f"{k}={v}")]
    ffmpeg("-i", raw, "-map", "0", "-c", "copy", "-movflags", "+faststart+use_metadata_tags", *meta, final)
    dur, kinds = probe(final)
    poster = to_jpeg(final, os.path.join(tmp, "poster.jpg"), dur / 3)

    manifest = {
        "schema": "open-media-mcp/provenance/v1",
        "job_id": job_id, "created_at": job["created_at"], "completed_at": now(),
        "caller": {"tenant": job.get("tenant"), "principal": job.get("principal")},
        "backend": {"name": backend.name, "license": backend.license, "notice": backend.notice,
                    "model": backend.model_file(spec)},
        "request": {k: v for k, v in spec.items() if k != "prompt"},
        "recipe": spec.get("recipe"),
        "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest(),
        "prompt_rewrite": ({**prompt_info, "rewritten_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest()}
                           if prompt_info else None),
        "inputs": prep_tool.inputs,
        "render": {"comfyui_commit": open("/opt/media/comfyui_commit.txt").read().strip()
                   if os.path.exists("/opt/media/comfyui_commit.txt") else None,
                   "frames": backend.frames(spec["seconds"]), "fps": backend.output_fps(spec),
                   "render_seconds": round(render_s, 1), "soundtrack_muxed": muxed, "finishing": finishing,
                   "worker": INSTANCE},
        "moderation": moderation,
        "identity": ident,
        "guardrail": job.get("guardrail"),
        "output": {"sha256": sha256(final), "bytes": os.path.getsize(final), "duration_s": round(dur, 3),
                   "has_audio": "audio" in kinds},
    }
    mpath = os.path.join(tmp, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)

    base = f"outputs/{job_id}/"
    s3.upload_file(final, E["MEDIA_BUCKET"], base + f"{job_id}.mp4", ExtraArgs={"ContentType": "video/mp4"})
    s3.upload_file(poster, E["MEDIA_BUCKET"], base + "poster.jpg", ExtraArgs={"ContentType": "image/jpeg"})
    s3.upload_file(mpath, E["MEDIA_BUCKET"], base + "manifest.json", ExtraArgs={"ContentType": "application/json"})
    return {
        "output": {"video_key": base + f"{job_id}.mp4", "poster_key": base + "poster.jpg",
                   "manifest_key": base + "manifest.json", "duration_s": round(dur, 3),
                   "has_audio": "audio" in kinds, "sha256": manifest["output"]["sha256"]},
        "timings": {"render_s": round(render_s, 1), "total_s": round(time.time() - t_start, 1)},
        "moderation": moderation,
        **({"identity": ident} if ident else {}),
        "cost": job_cost(time.time() - t_start),
    }


def finish_audio(job, spec, backend, hist, tmp, render_s, t_start, inputs=(), prompt_info=None):
    """Audio backends: fetch ComfyUI's file, ship a lossless WAV + an MP3, tag both, write the manifest."""
    job_id = job["job_id"]
    files = [f for out in hist["outputs"].values() for v in out.values() if isinstance(v, list)
             for f in v if isinstance(f, dict) and str(f.get("filename", "")).endswith((".flac", ".wav", ".mp3", ".opus"))]
    if not files:
        raise RuntimeError(f"no audio in outputs: {json.dumps(hist['outputs'])[:800]}")
    f = files[0]
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
    raw = os.path.join(tmp, "raw" + os.path.splitext(f["filename"])[1])
    with open(raw, "wb") as fh:
        fh.write(comfy_get(f"/view?{q}", timeout=600))
    update_job(job_id, phase="packaging")
    tags = {"comment": f"AI-generated by open-media-mcp {VERSION} ({backend.name}); job {job_id}",
            "ai_generated": "true", "generator": f"open-media-mcp/{VERSION}", "backend": backend.name, "job_id": job_id}
    if backend.notice:
        tags["copyright"] = backend.notice
    meta = [a for k, v in tags.items() for a in ("-metadata", f"{k}={v}")]
    wav = os.path.join(tmp, f"{job_id}.wav")
    mp3 = os.path.join(tmp, f"{job_id}.mp3")
    ffmpeg("-i", raw, "-c:a", "pcm_s16le", *meta, wav)
    ffmpeg("-i", raw, "-c:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3", *meta, mp3)
    dur, _ = probe(wav)
    manifest = {
        "schema": "open-media-mcp/provenance/v1",
        "job_id": job_id, "created_at": job["created_at"], "completed_at": now(),
        "caller": {"tenant": job.get("tenant"), "principal": job.get("principal")},
        "backend": {"name": backend.name, "license": backend.license, "notice": backend.notice, "model": backend.model_file(spec)},
        "request": {k: v for k, v in spec.items() if k != "prompt"},
        "recipe": spec.get("recipe"),
        "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest(),
        "prompt_rewrite": ({**prompt_info, "rewritten_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest()}
                           if prompt_info else None),
        "inputs": list(inputs),
        "render": {"comfyui_commit": open("/opt/media/comfyui_commit.txt").read().strip()
                   if os.path.exists("/opt/media/comfyui_commit.txt") else None,
                   "render_seconds": round(render_s, 1), "worker": INSTANCE},
        "guardrail": job.get("guardrail"),
        "output": {"sha256": sha256(wav), "bytes": os.path.getsize(wav), "duration_s": round(dur, 3), "mp3_sha256": sha256(mp3)},
    }
    mpath = os.path.join(tmp, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    base = f"outputs/{job_id}/"
    s3.upload_file(wav, E["MEDIA_BUCKET"], base + f"{job_id}.wav", ExtraArgs={"ContentType": "audio/wav"})
    s3.upload_file(mp3, E["MEDIA_BUCKET"], base + f"{job_id}.mp3", ExtraArgs={"ContentType": "audio/mpeg"})
    s3.upload_file(mpath, E["MEDIA_BUCKET"], base + "manifest.json", ExtraArgs={"ContentType": "application/json"})
    return {
        "output": {"audio_key": base + f"{job_id}.wav", "mp3_key": base + f"{job_id}.mp3", "manifest_key": base + "manifest.json",
                   "duration_s": round(dur, 3), "has_audio": True, "sha256": manifest["output"]["sha256"]},
        "timings": {"render_s": round(render_s, 1), "total_s": round(time.time() - t_start, 1)},
        "moderation": {"enabled": False, "note": "audio output is not frame-checked"},
        "cost": job_cost(time.time() - t_start),
    }


def finish_image(job, spec, backend, hist, tmp, render_s, t_start, inputs=(), prompt_info=None):
    """Image backends: fetch each PNG ComfyUI saved (sampler order), content-check them if on, tag them, write the manifest."""
    from PIL import Image, PngImagePlugin  # a ComfyUI requirement, so in this venv; only image jobs need it
    job_id = job["job_id"]
    files = [f for _, out in sorted(hist["outputs"].items()) for f in out.get("images", [])
             if isinstance(f, dict) and str(f.get("filename", "")).endswith(".png")]
    if not files:
        raise RuntimeError(f"no image in outputs: {json.dumps(hist['outputs'])[:800]}")
    raws = []
    for i, f in enumerate(files, 1):
        q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
        raws.append(os.path.join(tmp, f"raw_{i}.png"))
        with open(raws[-1], "wb") as fh:
            fh.write(comfy_get(f"/view?{q}", timeout=600))
    update_job(job_id, phase="checking and packaging")

    moderation = {"enabled": MODERATE}
    if MODERATE:
        hits = moderate_jpegs([to_jpeg(p, os.path.join(tmp, f"image_{i}.jpg")) for i, p in enumerate(raws, 1)])
        moderation.update(images_checked=len(raws), hits=hits)
        if hits:
            qkey = f"quarantine/{job_id}/"
            for i, p in enumerate(raws, 1):
                s3.upload_file(p, E["MEDIA_BUCKET"], f"{qkey}{job_id}_{i}.png")
            raise JobError("the render failed the output content check (kept in quarantine for review)", "BLOCKED",
                           {"moderation": moderation, "quarantine_key": qkey})

    tags = {"Comment": f"AI-generated by open-media-mcp {VERSION} ({backend.name}); job {job_id}",
            "ai_generated": "true", "generator": f"open-media-mcp/{VERSION}", "backend": backend.name, "job_id": job_id}
    if backend.notice:
        tags["Copyright"] = backend.notice
    base = f"outputs/{job_id}/"
    # one record per PNG, in output order: its seed, and what else tells the images of one job apart (h3-image: easyfix)
    meta = backend.image_meta(spec) if hasattr(backend, "image_meta") else [{"seed": spec["seed"] + i} for i in range(len(raws))]
    if len(meta) != len(raws):
        raise RuntimeError(f"{len(raws)} images came back for {len(meta)} asked")
    images = []
    for i, raw in enumerate(raws):
        name = f"{job_id}.png" if len(raws) == 1 else f"{job_id}_{i + 1}.png"
        path = os.path.join(tmp, name)
        info = PngImagePlugin.PngInfo()
        for k, v in {**tags, **{k: str(v) for k, v in meta[i].items()}}.items():
            info.add_text(k, v)
        with Image.open(raw) as im:
            im.save(path, pnginfo=info)
            w, h = im.size
        images.append({"key": base + name, "path": path, **meta[i], "width": w, "height": h,
                       "sha256": sha256(path), "bytes": os.path.getsize(path)})
    manifest = {
        "schema": "open-media-mcp/provenance/v1",
        "job_id": job_id, "created_at": job["created_at"], "completed_at": now(),
        "caller": {"tenant": job.get("tenant"), "principal": job.get("principal")},
        "backend": {"name": backend.name, "license": backend.license, "notice": backend.notice, "model": backend.model_file(spec)},
        "request": {k: v for k, v in spec.items() if k != "prompt"},
        "recipe": spec.get("recipe"),
        "prompt_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest(),
        "prompt_rewrite": ({**prompt_info, "rewritten_sha256": hashlib.sha256(spec["prompt"].encode()).hexdigest()}
                           if prompt_info else None),
        "inputs": list(inputs),
        "render": {"comfyui_commit": open("/opt/media/comfyui_commit.txt").read().strip()
                   if os.path.exists("/opt/media/comfyui_commit.txt") else None,
                   "render_seconds": round(render_s, 1), "worker": INSTANCE},
        "moderation": moderation,
        "guardrail": job.get("guardrail"),
        "output": {"images": [{k: v for k, v in im.items() if k != "path"} for im in images]},
    }
    mpath = os.path.join(tmp, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    for im in images:
        s3.upload_file(im["path"], E["MEDIA_BUCKET"], im["key"], ExtraArgs={"ContentType": "image/png"})
    s3.upload_file(mpath, E["MEDIA_BUCKET"], base + "manifest.json", ExtraArgs={"ContentType": "application/json"})
    return {
        "output": {"images": [{k: im[k] for k in ("key", "seed", "easyfix", "width", "height", "sha256") if k in im} for im in images],
                   "manifest_key": base + "manifest.json"},
        "timings": {"render_s": round(render_s, 1), "total_s": round(time.time() - t_start, 1)},
        "moderation": moderation,
        "cost": job_cost(time.time() - t_start),
    }


def handle(msg):
    body = json.loads(msg["Body"])
    job_id = body["job_id"]
    receipt = msg["ReceiptHandle"]
    attempt = int(msg.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item or item["status"] in FINAL:
        sqs.delete_message(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=receipt)
        return
    if not update_job(job_id, status="STARTING", phase="downloading inputs", attempt=attempt, worker=INSTANCE,
                      started_at=now()):
        sqs.delete_message(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=receipt)
        return
    created = datetime.fromisoformat(item["created_at"])
    metric("QueueWaitSeconds", (datetime.now(timezone.utc) - created).total_seconds(), "Seconds")
    audit(event="job_start", job_id=job_id, tenant=item.get("tenant"), attempt=attempt)
    status_file("preparing", job_id)

    hb = Heartbeat(receipt, job_id)
    hb.start()
    tmp = tempfile.mkdtemp(prefix=f"{job_id}_", dir="/opt/media/work")
    preempted = False
    try:
        result = run_job(item, hb, tmp)
        update_job(job_id, status="SUCCEEDED", phase="done", finished_at=now(), **result)
        sqs.delete_message(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=receipt)
        metric("JobSucceeded")
        metric("RenderSeconds", result["timings"]["render_s"], "Seconds")
        if result["cost"]["usd"] is not None:
            metric("JobCostUsd", result["cost"]["usd"], "None")
        audit(event="job_succeeded", job_id=job_id, **result["timings"])
        emit(item, "SUCCEEDED", {"output": result["output"]})
    except Cancelled:
        sqs.delete_message(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=receipt)
        metric("JobCancelled")
        audit(event="job_cancelled", job_id=job_id)
        emit(item, "CANCELLED")
    except Preempted:
        # Spot reclaim in ~2 min: hand the job straight back so another worker picks it up.
        update_job(job_id, status="QUEUED", phase="requeued after a spot interruption")
        sqs.change_message_visibility(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=receipt, VisibilityTimeout=0)
        audit(event="job_preempted", job_id=job_id)
        preempted = True
    except JobError as exc:
        update_job(job_id, status=exc.status, phase="done", error=str(exc), finished_at=now(), **exc.details)
        sqs.delete_message(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=receipt)
        metric("JobBlocked" if exc.status == "BLOCKED" else "JobFailed")
        audit(event="job_" + exc.status.lower(), job_id=job_id, error=str(exc), **exc.details)
        emit(item, exc.status, {"error": str(exc)})
    except Exception as exc:
        log.exception("job %s attempt %s", job_id, attempt)
        if attempt >= 3:
            # Leave the message: SQS dead-letters it on the next receive (and the DLQ alarm fires).
            update_job(job_id, status="FAILED", phase="done", error=str(exc)[:2000], finished_at=now())
            metric("JobFailed")
            emit(item, "FAILED", {"error": str(exc)[:500]})
        else:
            update_job(job_id, status="QUEUED", phase=f"retrying after attempt {attempt} failed", error=str(exc)[:2000])
            sqs.change_message_visibility(QueueUrl=E["MEDIA_QUEUE_URL"], ReceiptHandle=receipt, VisibilityTimeout=10)
        audit(event="job_error", job_id=job_id, attempt=attempt, error=str(exc)[:2000])
    finally:
        hb.stop_evt.set()
        shutil.rmtree(tmp, ignore_errors=True)
    if preempted:
        status_file("spot interruption, shutting down")
        time.sleep(180)


def queue_empty():
    a = sqs.get_queue_attributes(QueueUrl=E["MEDIA_QUEUE_URL"],
                                 AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
    return int(a["ApproximateNumberOfMessages"]) == 0 and int(a["ApproximateNumberOfMessagesNotVisible"]) == 0


def scale_in():
    status_file("idle, leaving the group")
    audit(event="scale_in", reason=f"idle {E['MEDIA_IDLE_MINUTES']} min")
    try:
        asg.terminate_instance_in_auto_scaling_group(InstanceId=INSTANCE["instance_id"], ShouldDecrementDesiredCapacity=True)
    except Exception as exc:
        log.warning("self-terminate failed (min size reached?): %s", exc)


BACKENDS = backend_registry.load([b for b in E["MEDIA_BACKENDS"].split(",") if b])


def main():
    os.makedirs("/opt/media/work", exist_ok=True)
    wait_for_comfy()
    idle_limit = float(E.get("MEDIA_IDLE_MINUTES", "20")) * 60
    last_work = time.time()
    status_file("ready")
    audit(event="worker_ready", backends=list(BACKENDS), **INSTANCE)
    last_status = time.time()
    while True:
        resp = sqs.receive_message(QueueUrl=E["MEDIA_QUEUE_URL"], MaxNumberOfMessages=1, WaitTimeSeconds=20,
                                   VisibilityTimeout=LEASE_SECONDS, AttributeNames=["ApproximateReceiveCount"])
        msgs = resp.get("Messages", [])
        if msgs:
            handle(msgs[0])
            last_work = time.time()
            status_file("ready")
            continue
        if time.time() - last_status > 60:
            status_file(f"ready, idle {int((time.time() - last_work) / 60)} min")
            last_status = time.time()
        if idle_limit > 0 and time.time() - last_work > idle_limit and queue_empty():
            scale_in()
            time.sleep(600)
            last_work = time.time()


if __name__ == "__main__":
    main()
