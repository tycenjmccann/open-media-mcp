"""Populate ComfyUI/models for the enabled backends: S3 cache first (fast, same region), then Hugging Face,
then back-fill the stack's own bucket so the next boot (or the next spot replacement) is warm.

  fetch_models.py --backends wan2.2,h3 --models /opt/media/ComfyUI/models --stage /opt/media/hf-stage
Env: MEDIA_MODEL_BUCKETS (first = writable stack bucket, rest = read-only seed caches), MEDIA_HF_TOKEN_SECRET.
"""
import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backends as registry  # noqa: E402


def s3_has(bucket, key):
    return subprocess.run(["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
                          capture_output=True).returncode == 0


def hf_token():
    name = os.environ.get("MEDIA_HF_TOKEN_SECRET")
    if not name:
        return None
    out = subprocess.run(["aws", "secretsmanager", "get-secret-value", "--secret-id", name, "--query", "SecretString",
                          "--output", "text"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def fetch(entry, models, stage, buckets, token):
    from huggingface_hub import hf_hub_download
    repo, file, subdir = entry
    name = os.path.basename(file)
    dest = os.path.join(models, subdir, name)
    key = f"models/{subdir}/{name}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return f"present  {key}"
    for b in buckets:
        if s3_has(b, key):
            subprocess.run(["aws", "s3", "cp", f"s3://{b}/{key}", dest, "--only-show-errors"], check=True)
            return f"s3       {key} ({b})"
    staged = hf_hub_download(repo, file, local_dir=os.path.join(stage, repo.replace("/", "__")), token=token)
    shutil.move(staged, dest)
    subprocess.run(["aws", "s3", "cp", dest, f"s3://{buckets[0]}/{key}", "--only-show-errors"], check=True)
    return f"hf->s3   {key}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--stage", required=True)
    a = ap.parse_args()
    buckets = [b for b in os.environ["MEDIA_MODEL_BUCKETS"].split(",") if b]
    entries = list(dict.fromkeys(e for b in registry.load(a.backends.split(",")).values() for e in b.models))
    token = hf_token()
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(fetch, e, a.models, a.stage, buckets, token): e for e in entries}
        for f in cf.as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as exc:  # keep going; report every bad file at once
                failed += 1
                print(f"FAILED   {futs[f]}: {exc}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
