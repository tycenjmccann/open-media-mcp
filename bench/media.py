"""Shared helpers for the bench scripts: ffmpeg, probing, frame decoding, suite and run-dir loading."""
import hashlib
import json
import os
import re
import shutil
import subprocess

import numpy as np


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()
FFPROBE = shutil.which("ffprobe")  # imageio-ffmpeg ships no ffprobe; probe() falls back to parsing `ffmpeg -i`
CLIP_EXTS = (".mp4", ".wav")


def run(args, binary=False):
    p = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", *args], capture_output=True)
    return p.stdout if binary else p.stderr.decode(errors="replace")


def _rate(s):
    num, _, den = str(s).partition("/")
    try:
        return float(num) / float(den or 1) if float(den or 1) else None
    except ValueError:
        return None


def probe(path):
    """{width, height, fps, frames, duration, audio: {sample_rate, channels, bitrate} | None}; video keys None for audio files."""
    info = {"width": None, "height": None, "fps": None, "frames": None, "duration": None, "audio": None}
    if FFPROBE:
        p = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                           capture_output=True, text=True)
        d = json.loads(p.stdout or "{}")
        info["duration"] = float(d.get("format", {}).get("duration") or 0) or None
        for s in d.get("streams", []):
            if s["codec_type"] == "video" and info["width"] is None and s.get("disposition", {}).get("attached_pic") != 1:
                info.update(width=s["width"], height=s["height"], fps=_rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate")),
                            frames=int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None)
            elif s["codec_type"] == "audio" and info["audio"] is None:
                br = s.get("bit_rate") or (d["format"].get("bit_rate") if len(d.get("streams", [])) == 1 else None)
                info["audio"] = {"sample_rate": int(s["sample_rate"]), "channels": s.get("channels"),
                                 "bitrate": int(br) if br else None}
        return info
    err = run(["-i", path])
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if m:
        info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    m = re.search(r"Stream #.*?Video:.*?(\d{2,5})x(\d{2,5}).*?([\d.]+) fps", err)
    if m:
        info.update(width=int(m[1]), height=int(m[2]), fps=float(m[3]))
    m = re.search(r"Stream #.*?Audio:.*?(\d+) Hz, ([^,]+)(?:,[^,]+){1,2}(?:, (\d+) kb/s)?", err)
    if m:
        ch = {"mono": 1, "stereo": 2, "5.1": 6, "7.1": 8}.get(m[2].split("(")[0].strip())
        if ch is None and m[2].split()[0].isdigit():
            ch = int(m[2].split()[0])
        info["audio"] = {"sample_rate": int(m[1]), "channels": ch, "bitrate": int(m[3]) * 1000 if m[3] else None}
    return info


def scaled_size(w, h, width=None, height=None):
    """Even-numbered output size for a fixed width or height, keeping the aspect ratio."""
    if width:
        return width, max(2, round(width * h / w / 2) * 2)
    return max(2, round(height * w / h / 2) * 2), height


def gray_frames(path, w, h, scale_flags="area"):
    """Yield every frame as an (h, w) uint8 luma array, scaled by ffmpeg."""
    p = subprocess.Popen([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", path, "-map", "0:v:0",
                          "-vf", f"scale={w}:{h}:flags={scale_flags},format=gray", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE)
    size = w * h
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        p.stdout.close()
        p.wait()


def jpeg_frame(path, t, long_edge=768):
    """One frame at time t as JPEG bytes, long edge scaled to long_edge."""
    vf = f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
    return run(["-loglevel", "error", "-ss", f"{max(0.0, t):.4f}", "-i", path, "-frames:v", "1", "-vf", vf,
                "-q:v", "3", "-f", "image2pipe", "-c:v", "mjpeg", "-"], binary=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_suite(path):
    """Case dicts keyed by id, in suite order."""
    if not path:
        return {}
    cases = json.load(open(path))["cases"]
    ids = [c["id"] for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"duplicate case ids in {path}: {sorted(dupes)}")
    return {c["id"]: c for c in cases}


def clips(run_dir):
    """{case_id: path} for every <case_id>.mp4 / .wav in a run dir."""
    out = {}
    for f in sorted(os.listdir(run_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in CLIP_EXTS and not f.startswith("."):
            out[stem] = os.path.join(run_dir, f)
    return out


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default
