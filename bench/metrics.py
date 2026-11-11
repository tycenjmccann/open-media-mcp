"""Objective per-clip metrics for one bench run dir.

  python bench/metrics.py out/bench/r1/ --suite bench/suite.json

Writes out/bench/r1/metrics.json keyed by case id. Clips whose file and expect_speech are unchanged
since the last run are not recomputed (--force to redo).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import media  # noqa: E402

MOTION_WIDTH = 320    # flow is measured at this width so resolution doesn't change the number
SHARP_HEIGHT = 720    # sharpness is measured at this height so an upscaled clip isn't rewarded for its pixel count
SHARP_SAMPLES = 12


# The model's own frame rate: frames in between (from interpolation) are softer and change less per frame, so
# sharpness and flicker are measured on the model's frames only.
NATIVE_FPS = {"wan2.2": 16, "h3": 24}


def video_metrics(path, info, native_fps=None):
    w, h = media.scaled_size(info["width"], info["height"], width=MOTION_WIDTH)
    prev, flow_mag, luma = None, [], []
    for f in media.gray_frames(path, w, h):
        luma.append(float(f.mean()))
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, f, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_mag.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev = f
    n = len(luma)
    fps = info["fps"] or (n / info["duration"] if info["duration"] else None)
    out = {"width": info["width"], "height": info["height"], "fps": round(fps, 3) if fps else None, "frames": n,
           "duration": round(info["duration"] or n / fps, 3) if (info["duration"] or fps) else None}
    # px/frame * frames/s = px/s, so a 32 fps interpolated clip scores the same motion as its 16 fps source
    out["motion"] = round(float(np.mean(flow_mag)) * fps, 3) if flow_mag and fps else None
    step = int(round(fps / native_fps)) if fps and native_fps and fps > native_fps * 1.5 else 1
    native = luma[::step]
    out["flicker"] = round(float(np.mean(np.abs(np.diff(native)))), 4) if len(native) > 1 else None

    sw, sh = media.scaled_size(info["width"], info["height"], height=SHARP_HEIGHT)
    own = list(range(0, n, step))
    pick = {own[i] for i in np.linspace(0, len(own) - 1, min(SHARP_SAMPLES, len(own))).round().astype(int)} if n else set()
    lap = [float(cv2.Laplacian(f, cv2.CV_64F).var())
           for i, f in enumerate(media.gray_frames(path, sw, sh, scale_flags="bicubic")) if i in pick]
    out["sharpness"] = round(float(np.mean(lap)), 2) if lap else None
    return out


def loudness(path):
    err = media.run(["-nostats", "-i", path, "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"])
    summary = err[err.rfind("Summary:"):]

    def grab(label):
        m = re.search(rf"{label}:\s+(-?[\d.]+|-inf)", summary)
        return None if not m or m[1] == "-inf" else float(m[1])
    return {"lufs": grab("I"), "true_peak_dbfs": grab("Peak"), "lra": grab("LRA")}


def clip_metrics(path, native_fps=None):
    info = media.probe(path)
    out = video_metrics(path, info, native_fps) if info["width"] else {"duration": round(info["duration"], 3) if info["duration"] else None}
    if info["audio"]:
        out["audio"] = {**info["audio"], **loudness(path)}
    return out


def normalize(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text).replace("'", "")
    return text.split()


def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return round(d[len(h)] / len(r), 4)


def transcribe_child(paths, model_name):
    """Runs in a subprocess (see transcribe) and prints {path: text} as JSON."""
    def pcm16k(p):  # decode with ffmpeg ourselves: faster-whisper's PyAV path breaks on newer av releases
        return np.frombuffer(media.run(["-loglevel", "error", "-i", p, "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                                        "-f", "f32le", "-"], binary=True), np.float32).copy()

    texts = {}
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        engine = f"faster-whisper {model_name}"
    except Exception as exc:  # model download blocked (corp network) -> openai-whisper with a cached ~/.cache/whisper model
        print(f"faster-whisper unavailable ({type(exc).__name__}: {exc}); trying openai-whisper", file=sys.stderr)
        import whisper
        model, engine = whisper.load_model(model_name), f"openai-whisper {model_name}"
    for p in paths:
        if engine.startswith("faster"):
            segs, _ = model.transcribe(pcm16k(p), language="en", beam_size=5, condition_on_previous_text=False)
            texts[p] = " ".join(s.text.strip() for s in segs).strip()
        else:
            texts[p] = model.transcribe(pcm16k(p), language="en", fp16=False)["text"].strip()
    print(json.dumps({"engine": engine, "texts": texts}))


def transcribe(paths, model_name):
    # Separate process: faster-whisper's PyAV and OpenCV bundle clashing copies of libavdevice on macOS.
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--_transcribe", json.dumps(paths),
                        "--whisper-model", model_name], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"transcription failed:\n{p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--suite", help="suite JSON (for expect_speech)")
    ap.add_argument("--whisper-model", default="small.en")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute every clip")
    ap.add_argument("--_transcribe", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._transcribe:
        return transcribe_child(json.loads(a._transcribe), a.whisper_model)
    if not a.run_dir:
        ap.error("run_dir is required")

    suite = media.load_suite(a.suite)
    clips = media.clips(a.run_dir)
    out_path = os.path.join(a.run_dir, "metrics.json")
    old = {} if a.force else media.load_json(out_path, {})

    def source(cid, path):
        st = os.stat(path)
        return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "expect_speech": suite.get(cid, {}).get("expect_speech"),
                "metrics_v": 2}

    todo = [cid for cid, p in clips.items() if old.get(cid, {}).get("source") != source(cid, p)]
    result = {cid: old[cid] for cid in clips if cid not in todo}
    with ThreadPoolExecutor(a.workers) as pool:
        for cid, m in zip(todo, pool.map(lambda c: clip_metrics(clips[c], NATIVE_FPS.get(suite.get(c, {}).get("backend"))), todo)):
            result[cid] = {**m, "source": source(cid, clips[cid])}
            print(f"{cid}: motion {m.get('motion')} px/s, sharpness {m.get('sharpness')}, flicker {m.get('flicker')}, "
                  f"LUFS {(m.get('audio') or {}).get('lufs')}", flush=True)

    speech = [cid for cid in todo if isinstance(suite.get(cid, {}).get("expect_speech"), str)
              and suite[cid]["expect_speech"].strip()]
    for cid in [c for c in speech if not result[c].get("audio")]:
        speech.remove(cid)
        result[cid]["speech"] = {"expected": suite[cid]["expect_speech"], "transcript": "", "wer": 1.0, "engine": "no audio stream"}
    if speech:
        t = transcribe([clips[c] for c in speech], a.whisper_model)
        for cid in speech:
            hyp, ref = t["texts"][clips[cid]], suite[cid]["expect_speech"]
            result[cid]["speech"] = {"expected": ref, "transcript": hyp, "wer": wer(ref, hyp), "engine": t["engine"]}
            print(f"{cid}: WER {result[cid]['speech']['wer']} ({hyp!r})", flush=True)

    json.dump({cid: result[cid] for cid in clips}, open(out_path, "w"), indent=1)
    print(f"{len(clips)} clips ({len(todo)} measured) -> {out_path}")


if __name__ == "__main__":
    main()
