"""open-media-mcp tools: the AgentCore Gateway Lambda target.

Gateway invokes this function once per MCP tool call: the event is the tool arguments, the tool name arrives in
context.client_context.custom["bedrockAgentCoreToolName"] as "<target>___<tool>". The REQUEST interceptor adds a
verified `_caller` argument ({tenant, principal, auth}) that clients cannot forge (it overwrites any they send).

Every tool returns in well under a second: generate_video only validates, screens the prompt, records the job and
enqueues it. The GPU worker does the rest. An EventBridge schedule also invokes this function ({"reconcile": true})
to make sure queued work always has a GPU coming.
"""
import hashlib
import json
import math
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

E = os.environ
cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"})
ddb = boto3.resource("dynamodb", config=cfg)
table = ddb.Table(E["MEDIA_TABLE"])
sqs = boto3.client("sqs", config=cfg)
# SigV4 is required for presigned URLs on SSE-KMS objects.
s3 = boto3.client("s3", config=cfg.merge(Config(signature_version="s3v4")), region_name=os.environ.get("AWS_REGION"))
asg = boto3.client("autoscaling", config=cfg)
bedrock = boto3.client("bedrock-runtime", config=cfg)
cloudwatch = boto3.client("cloudwatch", config=cfg)
ec2 = boto3.client("ec2", config=cfg)

with open(os.path.join(os.path.dirname(__file__), "backends.json")) as fh:
    ALL_BACKENDS = json.load(fh)
ENABLED = [b for b in E.get("MEDIA_BACKENDS", "wan2.2").split(",") if b]
BACKENDS = {k: v for k, v in ALL_BACKENDS.items() if k in ENABLED}
DEFAULT_BACKEND = ENABLED[0]
FINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}
ACTIVE = {"QUEUED", "STARTING", "RUNNING"}
UPLOAD_TYPES = {
    "image/png", "image/jpeg", "image/webp",
    "audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp4", "audio/aac", "audio/flac", "audio/ogg",
    "video/mp4", "video/quicktime", "video/webm",
}
PRESIGN_TTL = int(E.get("MEDIA_PRESIGN_TTL", "3600"))
COLD_START_S = 330


class ToolError(Exception):
    pass


def now():
    return datetime.now(timezone.utc)


def jsonable(v):
    if isinstance(v, Decimal):
        return int(v) if v == v.to_integral_value() else float(v)
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jsonable(x) for x in v]
    return v


def ddb_safe(v):
    return json.loads(json.dumps(v, default=str), parse_float=Decimal)


def tenant_prefix(tenant):
    return hashlib.sha256(tenant.encode()).hexdigest()[:20]


# --------------------------------------------------------------------------- capacity
def fleet():
    g = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[E["MEDIA_ASG"]])["AutoScalingGroups"][0]
    q = sqs.get_queue_attributes(QueueUrl=E["MEDIA_QUEUE_URL"], AttributeNames=[
        "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
    workers = []
    try:
        listing = s3.list_objects_v2(Bucket=E["MEDIA_BUCKET"], Prefix="status/workers/").get("Contents", [])
        live = {i["InstanceId"] for i in g["Instances"]}
        for obj in listing:
            iid = obj["Key"].rsplit("/", 1)[-1].removesuffix(".json")
            if obj["Key"].endswith(".json") and iid in live:
                workers.append(json.loads(s3.get_object(Bucket=E["MEDIA_BUCKET"], Key=obj["Key"])["Body"].read()))
    except ClientError:
        pass
    dist = g.get("MixedInstancesPolicy", {}).get("InstancesDistribution", {})
    return {
        "desired": g["DesiredCapacity"], "max": g["MaxSize"],
        "instances": [{"id": i["InstanceId"], "type": i.get("InstanceType"), "state": i["LifecycleState"],
                       "az": i["AvailabilityZone"]} for i in g["Instances"]],
        "workers": workers,
        "purchase": "on-demand" if dist.get("OnDemandPercentageAboveBaseCapacity", 0) >= 100 else "spot",
        "queued": int(q["ApproximateNumberOfMessages"]), "in_flight": int(q["ApproximateNumberOfMessagesNotVisible"]),
    }


def scale_out(f=None):
    """One GPU per outstanding job, capped at MEDIA_MAX_INSTANCES. Never scales in (workers do that themselves)."""
    f = f or fleet()
    want = min(int(E["MEDIA_MAX_INSTANCES"]), f["queued"] + f["in_flight"])
    if want > f["desired"]:
        asg.set_desired_capacity(AutoScalingGroupName=E["MEDIA_ASG"], DesiredCapacity=want, HonorCooldown=False)
        f["desired"] = want
    return f


def ready_workers(f):
    return sum(1 for w in f["workers"] if str(w.get("phase", "")).startswith(("ready", "rendering", "preparing")))


def estimate(backend, seconds, quality):
    r = BACKENDS[backend]["render_estimate"]
    return int(r[quality] * seconds ** r["exponent"])


# --------------------------------------------------------------------------- tools
def get_capabilities(args, caller):
    f = fleet()
    return {
        "backends": {k: {kk: vv for kk, vv in v.items() if kk != "render_estimate"} |
                     {"render_estimate_s": {q: estimate(k, v["default_seconds"], q) for q in v["qualities"]}}
                     for k, v in BACKENDS.items()},
        "default_backend": DEFAULT_BACKEND,
        "prompt_tags": "Reference media are tagged in the prompt by upload order: refs -> <Picture 1..n>; first_frame, "
                       "keyframes, last_frame continue the <Picture> numbering; ref_videos -> <Video n>; "
                       "ref_audios -> <Audio n> (h3 only).",
        "safety": {"prompt_guardrail": bool(E.get("MEDIA_GUARDRAIL_ID")),
                   "output_moderation": E.get("MEDIA_MODERATION", "true") == "true"},
        "limits": {"daily_jobs_per_caller": int(E.get("MEDIA_DAILY_JOB_LIMIT", "0")) or None,
                   "max_queued_jobs": int(E["MEDIA_MAX_QUEUED"]), "url_ttl_s": PRESIGN_TTL},
        "gpu": {"ready_workers": ready_workers(f), "desired": f["desired"], "max": f["max"], "purchase": f["purchase"],
                "cold_start_s": COLD_START_S, "idle_scale_in_min": int(E["MEDIA_IDLE_MINUTES"])},
        "caller": caller,
    }


def create_upload(args, caller):
    ctype = str(args.get("content_type", "")).lower()
    if ctype not in UPLOAD_TYPES:
        raise ToolError(f"content_type must be one of {sorted(UPLOAD_TYPES)}")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(str(args.get("filename") or "file")))[-80:] or "file"
    key = f"inputs/{tenant_prefix(caller['tenant'])}/{uuid.uuid4().hex[:12]}/{name}"
    url = s3.generate_presigned_url("put_object", Params={"Bucket": E["MEDIA_BUCKET"], "Key": key, "ContentType": ctype},
                                    ExpiresIn=900)
    return {"key": key, "upload_url": url, "method": "PUT", "headers": {"Content-Type": ctype}, "expires_in_s": 900,
            "curl": f"curl -sf -X PUT -H 'Content-Type: {ctype}' --upload-file <file> '<upload_url>'"}


def _num(args, k, lo=None, hi=None):
    if args.get(k) is None:
        return None
    try:
        v = float(args[k])
    except (TypeError, ValueError):
        raise ToolError(f"{k} must be a number")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ToolError(f"{k} must be between {lo} and {hi}")
    return v


def _key(k, caller, what):
    if not isinstance(k, str) or not k.startswith(f"inputs/{tenant_prefix(caller['tenant'])}/"):
        raise ToolError(f"{what}: use a key returned by create_upload for this caller")
    try:
        s3.head_object(Bucket=E["MEDIA_BUCKET"], Key=k)
    except ClientError:
        raise ToolError(f"{what}: {k} has not been uploaded yet (PUT the file to the upload_url first)")
    return k


def _window(item, caller, what):
    if not isinstance(item, dict) or "key" not in item:
        raise ToolError(f"{what}: each item needs a key")
    out = {"key": _key(item["key"], caller, what)}
    for k in ("from", "to"):
        if item.get(k) is not None:
            out[k] = _num(item, k, 0, 36000)
    if "from" in out and "to" in out and out["to"] <= out["from"]:
        raise ToolError(f"{what}: to must be after from")
    return out


def normalize(args, caller):
    backend = args.get("backend") or DEFAULT_BACKEND
    if backend not in BACKENDS:
        raise ToolError(f"backend must be one of {list(BACKENDS)}")
    b = BACKENDS[backend]
    prompt = str(args.get("prompt") or "").strip()
    if not prompt or len(prompt) > 4000:
        raise ToolError("prompt is required (max 4000 characters)")
    aspect = args.get("aspect") or b["aspects"][0 if backend == "wan2.2" else 1]
    if aspect not in b["aspects"]:
        raise ToolError(f"aspect must be one of {b['aspects']}")
    quality = args.get("quality") or "full"
    if quality not in b["qualities"]:
        raise ToolError(f"quality must be one of {b['qualities']}")
    model = args.get("model")
    if model and model not in b["models"]:
        raise ToolError(f"model must be one of {b['models']} for {backend}")
    lim = b["limits"]
    spec = {"backend": backend, "prompt": prompt, "aspect": aspect, "quality": quality, "model": model,
            "seed": int(args["seed"]) if args.get("seed") is not None else random.randint(1, 2**31 - 1)}
    if args.get("negative_prompt"):
        spec["negative_prompt"] = str(args["negative_prompt"])[:2000]
    for field in ("refs", "ref_videos", "ref_audios", "keyframes"):
        items = args.get(field) or []
        if not isinstance(items, list):
            raise ToolError(f"{field} must be a list")
        if len(items) > lim[field]:
            raise ToolError(f"{backend} accepts at most {lim[field]} {field}" + (" (use backend h3 for references)" if not lim[field] else ""))
    spec["refs"] = [{"key": _key(i.get("key") if isinstance(i, dict) else i, caller, "refs")} for i in args.get("refs") or []]
    spec["ref_videos"] = [_window(i, caller, "ref_videos") for i in args.get("ref_videos") or []]
    spec["ref_audios"] = [_window(i, caller, "ref_audios") for i in args.get("ref_audios") or []]
    spec["keyframes"] = []
    for i in args.get("keyframes") or []:
        if not isinstance(i, dict) or i.get("at") is None:
            raise ToolError("keyframes: each needs key and at")
        spec["keyframes"].append({"key": _key(i.get("key"), caller, "keyframes"), "at": _num(i, "at", -15, 15)})
    for k in ("first_frame", "last_frame"):
        if args.get(k):
            if not b[k]:
                raise ToolError(f"{backend} does not take {k}")
            spec[k] = _key(args[k], caller, k)
    if spec.get("last_frame") and b["last_frame"] == "needs first_frame" and not spec.get("first_frame"):
        raise ToolError(f"{backend}: last_frame needs a first_frame too")
    st = args.get("soundtrack")
    soundtrack_len = None
    if st:
        w = _window(st, caller, "soundtrack")
        w["at"] = _num(st, "at", 0, 15) or 0.0
        w["keep_original"] = bool(st.get("keep_original", True))
        spec["soundtrack"] = w
        if "from" in w and "to" in w:
            soundtrack_len = w["to"] - w["from"]
    lo, hi = b["seconds"]
    seconds = _num(args, "seconds", lo, hi)
    if seconds is None:
        seconds = min(hi, max(lo, (spec.get("soundtrack", {}).get("at", 0) + soundtrack_len))) if soundtrack_len else b["default_seconds"]
    spec["seconds"] = round(seconds, 3)
    return spec


def screen_prompt(prompt):
    gid = E.get("MEDIA_GUARDRAIL_ID")
    if not gid:
        return {"checked": False}
    r = bedrock.apply_guardrail(guardrailIdentifier=gid, guardrailVersion=E.get("MEDIA_GUARDRAIL_VERSION") or "DRAFT",
                                source="INPUT", content=[{"text": {"text": prompt}}])
    hits = [f"{x.get('type')}:{x.get('confidence')}" for a in r.get("assessments", [])
            for x in a.get("contentPolicy", {}).get("filters", []) if x.get("action") == "BLOCKED"]
    return {"checked": True, "action": r["action"], "filters": hits}


def take_quota(tenant):
    limit = int(E.get("MEDIA_DAILY_JOB_LIMIT", "0"))
    if limit <= 0:
        return
    day = now().strftime("%Y-%m-%d")
    try:
        table.update_item(Key={"job_id": f"quota#{tenant}#{day}"}, UpdateExpression="ADD n :one SET expires_at = :exp",
                          ConditionExpression="attribute_not_exists(n) OR n < :lim",
                          ExpressionAttributeValues={":one": 1, ":lim": limit,
                                                     ":exp": int((now() + timedelta(days=2)).timestamp())})
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise ToolError(f"daily job limit reached ({limit} per caller per UTC day)")
        raise


def generate_video(args, caller):
    spec = normalize(args, caller)
    tenant = caller["tenant"]
    idem = args.get("idempotency_key")
    job_id = ("j-" + hashlib.sha256(f"{tenant}|{idem}".encode()).hexdigest()[:24]) if idem else \
        f"j-{now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    if idem:
        existing = table.get_item(Key={"job_id": job_id}).get("Item")
        if existing:
            return {"job_id": job_id, "status": existing["status"], "deduplicated": True}
    labels = {str(k)[:64]: str(v)[:256] for k, v in (args.get("labels") or {}).items()} if isinstance(args.get("labels"), dict) else {}
    item = {
        "job_id": job_id, "tenant": tenant, "principal": caller.get("principal"), "auth": caller.get("auth"),
        "created_at": now().isoformat(timespec="seconds"), "backend": spec["backend"],
        "prompt_preview": spec["prompt"][:160], "spec": json.dumps(spec), "labels": labels,
        "expires_at": int((now() + timedelta(days=int(E["MEDIA_JOB_RETENTION_DAYS"]))).timestamp()),
    }

    screen = screen_prompt(spec["prompt"])
    item["guardrail"] = screen
    if screen.get("action") == "GUARDRAIL_INTERVENED":
        item.update(status="BLOCKED", phase="done", error="prompt blocked by the content guardrail")
        table.put_item(Item=ddb_safe(item))
        print(json.dumps({"event": "prompt_blocked", "job_id": job_id, "tenant": tenant, "filters": screen["filters"]}))
        raise ToolError(f"prompt blocked by the content guardrail ({', '.join(screen['filters']) or 'policy'}); job {job_id}")

    f = fleet()
    if f["queued"] >= int(E["MEDIA_MAX_QUEUED"]):
        raise ToolError(f"the queue is full ({f['queued']} jobs); try again later")
    take_quota(tenant)
    item.update(status="QUEUED", phase="waiting for a GPU")
    try:
        table.put_item(Item=ddb_safe(item), ConditionExpression="attribute_not_exists(job_id)")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return {"job_id": job_id, "status": "QUEUED", "deduplicated": True}
        raise
    sqs.send_message(QueueUrl=E["MEDIA_QUEUE_URL"], MessageBody=json.dumps({"job_id": job_id}))
    f["queued"] += 1
    f = scale_out(f)
    render = estimate(spec["backend"], spec["seconds"], spec["quality"])
    warm = ready_workers(f) > 0
    ahead = f["queued"] - 1 + f["in_flight"]
    eta = render + (0 if warm else COLD_START_S) + ahead * render // max(1, f["desired"])
    print(json.dumps({"event": "job_queued", "job_id": job_id, "tenant": tenant, "backend": spec["backend"],
                      "seconds": spec["seconds"], "quality": spec["quality"]}))
    out = {"job_id": job_id, "status": "QUEUED", "backend": spec["backend"], "seconds": spec["seconds"],
           "aspect": spec["aspect"], "quality": spec["quality"], "seed": spec["seed"], "eta_s": int(eta),
           "gpu": "ready" if warm else "starting (cold start about 5 min)", "poll": "get_job every 30-60 s"}
    if BACKENDS[spec["backend"]]["notice"]:
        out["notice"] = BACKENDS[spec["backend"]]["notice"]
    return out


def _own(job_id, caller):
    if not isinstance(job_id, str) or not job_id.startswith("j-"):
        raise ToolError("job_id is required")
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item or item.get("tenant") != caller["tenant"]:
        raise ToolError(f"no job {job_id} for this caller")
    return item


def _url(key):
    return s3.generate_presigned_url("get_object", Params={"Bucket": E["MEDIA_BUCKET"], "Key": key}, ExpiresIn=PRESIGN_TTL)


def get_job(args, caller):
    item = jsonable(_own(args.get("job_id"), caller))
    spec = json.loads(item.pop("spec"))
    out = {k: item.get(k) for k in ("job_id", "status", "phase", "backend", "created_at", "started_at",
                                    "render_started_at", "finished_at", "attempt", "error", "timings", "labels",
                                    "moderation", "quarantine_key", "cost") if item.get(k) is not None}
    out["request"] = {k: spec.get(k) for k in ("aspect", "seconds", "quality", "model", "seed") if spec.get(k) is not None}
    if item["status"] == "SUCCEEDED":
        o = item["output"]
        out["output"] = {"video_url": _url(o["video_key"]), "poster_url": _url(o["poster_key"]),
                         "manifest_url": _url(o["manifest_key"]), "duration_s": o["duration_s"],
                         "has_audio": o["has_audio"], "sha256": o["sha256"], "expires_in_s": PRESIGN_TTL,
                         "download": "curl -sfo clip.mp4 '<video_url>'"}
        notice = BACKENDS.get(item["backend"], ALL_BACKENDS.get(item["backend"], {})).get("notice")
        if notice:
            out["notice"] = notice
    elif item["status"] in ACTIVE:
        out["elapsed_s"] = int((now() - datetime.fromisoformat(item["created_at"])).total_seconds())
    return out


def list_jobs(args, caller):
    limit = max(1, min(50, int(args.get("limit") or 10)))
    want = args.get("status")
    kw = {"IndexName": "by_tenant", "KeyConditionExpression": "tenant = :t", "ScanIndexForward": False,
          "ExpressionAttributeValues": {":t": caller["tenant"]}, "Limit": limit if not want else 200}
    items = table.query(**kw)["Items"]
    if want:
        items = [i for i in items if i.get("status") == want]
    return {"jobs": [jsonable({**{k: i.get(k) for k in ("job_id", "status", "phase", "backend", "created_at", "prompt_preview", "labels")},
                               "cost_usd": (i.get("cost") or {}).get("usd")}) for i in items[:limit]]}


def cancel_job(args, caller):
    item = _own(args.get("job_id"), caller)
    try:
        table.update_item(Key={"job_id": item["job_id"]},
                          UpdateExpression="SET #s = :c, phase = :p, finished_at = :t",
                          ConditionExpression="#s IN (:q, :st, :r)",
                          ExpressionAttributeNames={"#s": "status"},
                          ExpressionAttributeValues={":c": "CANCELLED", ":p": "cancelled by caller", ":t": now().isoformat(timespec="seconds"),
                                                     ":q": "QUEUED", ":st": "STARTING", ":r": "RUNNING"})
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return {"job_id": item["job_id"], "status": item["status"], "cancelled": False, "reason": "already final"}
        raise
    return {"job_id": item["job_id"], "status": "CANCELLED", "cancelled": True}


def gpu_capacity(args, caller):
    action = args.get("action") or "status"
    f = fleet()
    if action == "warm":
        if f["desired"] < 1:
            asg.set_desired_capacity(AutoScalingGroupName=E["MEDIA_ASG"], DesiredCapacity=1, HonorCooldown=False)
            f["desired"] = 1
        f["note"] = f"a worker is starting; it stays up {E['MEDIA_IDLE_MINUTES']} idle minutes"
    elif action == "release":
        if f["queued"] or f["in_flight"]:
            raise ToolError(f"{f['queued']} queued and {f['in_flight']} running jobs; cancel them first")
        asg.set_desired_capacity(AutoScalingGroupName=E["MEDIA_ASG"], DesiredCapacity=0, HonorCooldown=False)
        f["desired"] = 0
        f["note"] = "scaling to zero"
    elif action in ("use_on_demand", "use_spot"):
        asg.update_auto_scaling_group(AutoScalingGroupName=E["MEDIA_ASG"], MixedInstancesPolicy={"InstancesDistribution": {
            "OnDemandPercentageAboveBaseCapacity": 100 if action == "use_on_demand" else 0}})
        f["purchase"] = "on-demand" if action == "use_on_demand" else "spot"
        f["note"] = "applies to the next instance launched (a redeploy restores the cdk.json setting)"
    elif action != "status":
        raise ToolError("action must be status, warm, release, use_on_demand or use_spot")
    if action != "status":
        print(json.dumps({"event": "capacity_" + action, "tenant": caller["tenant"], "principal": caller.get("principal")}))
    f["ready_workers"] = ready_workers(f)
    return f


def cost_report(args, caller):
    """Chargeback: exact per-job GPU cost for this caller, grouped by a label, plus the fleet's total instance time."""
    days = max(1, min(90, int(args.get("days") or 7)))
    group = args.get("group_by_label")
    since = now() - timedelta(days=days)
    kw = {"IndexName": "by_tenant", "KeyConditionExpression": "tenant = :t AND created_at >= :s",
          "ExpressionAttributeValues": {":t": caller["tenant"], ":s": since.isoformat(timespec="seconds")}}
    items = []
    while True:
        r = table.query(**kw)
        items += r["Items"]
        if "LastEvaluatedKey" not in r:
            break
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]
    items = jsonable(items)
    by = {}
    total = {"jobs": 0, "succeeded": 0, "usd": 0.0, "gpu_seconds": 0.0, "video_seconds": 0.0}
    prices = []
    for i in items:
        c = i.get("cost") or {}
        key = (i.get("labels") or {}).get(group, "(none)") if group else i.get("backend", "?")
        b = by.setdefault(key, {"jobs": 0, "succeeded": 0, "usd": 0.0, "gpu_seconds": 0.0, "video_seconds": 0.0})
        for t in (b, total):
            t["jobs"] += 1
            if i.get("status") == "SUCCEEDED":
                t["succeeded"] += 1
                t["video_seconds"] += (i.get("output") or {}).get("duration_s") or 0
            t["usd"] += c.get("usd") or 0
            t["gpu_seconds"] += c.get("gpu_seconds") or 0
        if c.get("usd_per_hour"):
            prices.append(c["usd_per_hour"])
    for t in [total, *by.values()]:
        t["usd"] = round(t["usd"], 4)
        t["usd_per_video_second"] = round(t["usd"] / t["video_seconds"], 5) if t["video_seconds"] else None
    # Fleet: every instance-hour the group ran (boot, render and idle), from the group's CloudWatch metric.
    span = (now() - since).total_seconds()
    period = max(60, math.ceil(span / 1440 / 60) * 60)
    pts = cloudwatch.get_metric_statistics(
        Namespace="AWS/AutoScaling", MetricName="GroupInServiceInstances",
        Dimensions=[{"Name": "AutoScalingGroupName", "Value": E["MEDIA_ASG"]}],
        StartTime=since, EndTime=now(), Period=period, Statistics=["Average"])["Datapoints"]
    inst_hours = sum(p["Average"] for p in pts) * period / 3600
    price = sum(prices) / len(prices) if prices else None
    if price is None:
        try:
            h = ec2.describe_spot_price_history(InstanceTypes=[E["MEDIA_INSTANCE_TYPES"].split(",")[0]],
                                                ProductDescriptions=["Linux/UNIX"], MaxResults=5)["SpotPriceHistory"]
            price = min(float(x["SpotPrice"]) for x in h)
        except ClientError:
            pass
    fleet = {"instance_hours": round(inst_hours, 3), "usd_per_hour": price,
             "usd": round(inst_hours * price, 4) if price else None,
             "note": "whole fleet, all callers: boot + render + idle time; jobs.usd is the part attributed to your jobs"}
    return {"days": days, "group_by": group or "backend", "jobs": total, "groups": by, "fleet": fleet}


TOOLS = {f.__name__: f for f in (get_capabilities, create_upload, generate_video, get_job, list_jobs, cancel_job, gpu_capacity, cost_report)}


def reconcile():
    """Scheduled: queued jobs must always have capacity coming (covers a worker that scaled in during a submit)."""
    f = fleet()
    in_service = sum(1 for i in f["instances"] if i["state"] in ("InService", "Pending", "Pending:Wait", "Pending:Proceed"))
    if (f["queued"] or f["in_flight"]) and in_service < min(int(E["MEDIA_MAX_INSTANCES"]), f["queued"] + f["in_flight"]):
        want = min(int(E["MEDIA_MAX_INSTANCES"]), f["queued"] + f["in_flight"])
        asg.set_desired_capacity(AutoScalingGroupName=E["MEDIA_ASG"], DesiredCapacity=max(want, f["desired"]), HonorCooldown=False)
        print(json.dumps({"event": "reconcile_scale_out", "desired": want, "queued": f["queued"]}))
    return {"ok": True}


def lambda_handler(event, context):
    if isinstance(event, dict) and event.get("reconcile"):
        return reconcile()
    raw = ""
    cc = getattr(context, "client_context", None)
    if cc and getattr(cc, "custom", None):
        raw = cc.custom.get("bedrockAgentCoreToolName", "")
    tool = raw.split("___")[-1]
    args = dict(event or {})
    caller = args.pop("_caller", None) or {"tenant": "unverified", "principal": "unknown", "auth": "none"}
    fn = TOOLS.get(tool)
    if not fn:
        return {"ok": False, "error": f"unknown tool {raw!r}"}
    t0 = time.time()
    try:
        out = fn(args, caller)
        return {"ok": True, **jsonable(out)}
    except ToolError as e:
        return {"ok": False, "error": str(e)}
    finally:
        print(json.dumps({"event": "tool_call", "tool": tool, "tenant": caller.get("tenant"),
                          "principal": caller.get("principal"), "ms": int((time.time() - t0) * 1000)}))
