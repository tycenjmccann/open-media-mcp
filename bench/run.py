"""Render a bench suite with one recipe through the MCP gateway (examples/batch.py does the submit/poll/download).

  python bench/run.py bench/suite.json --recipe r1 --name r1 [--tuning '{"k": v}'] [--only id1,id2] [--dry-run]

Writes out/bench/<name>/<case_id>.mp4 (.wav for audio cases), jobs.json (what was submitted) and results.json
(status, timings, cost per job). Needs GATEWAY_URL and AWS credentials, like examples/mcp_client.py.
Re-running the same --name resumes: finished cases are skipped and the idempotency key returns in-flight jobs.
"""
import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "examples"))
import batch  # noqa: E402
import media  # noqa: E402
from mcp_client import upload as upload_file  # noqa: E402

SUITE_ONLY = {"id", "expect_speech", "notes"}       # bench metadata, not generate_video arguments
LISTS = ("refs", "keyframes", "ref_audios", "ref_videos")  # arrays of {"key": ...}
DONE = ("SUCCEEDED", "QUEUED", "STARTING", "RUNNING")       # batch.py skips these on a re-run


def audio_backends():
    try:
        b = json.load(open(os.path.join(ROOT, "lambda", "tools", "backends.json")))
        return {k for k, v in b.items() if v.get("kind") == "audio"}
    except OSError:
        return {"stable-audio-3"}


def file_slots(args):
    """(container, key) for every argument that holds an upload key (or a local path to upload)."""
    for k in ("first_frame", "last_frame"):
        if isinstance(args.get(k), str):
            yield args, k
    if isinstance(args.get("soundtrack"), dict):
        yield args["soundtrack"], "key"
    for k in LISTS:
        for item in args.get(k) or []:
            yield item, "key"


def resolve(value, suite_dir, where):
    if value.startswith("inputs/"):  # already an upload key
        return value
    for base in (os.getcwd(), suite_dir):
        p = os.path.join(base, os.path.expanduser(value))
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise SystemExit(f"{where}: file not found: {value}")


def build_jobs(suite_path, recipe, tuning, name, only=None):
    suite = media.load_suite(suite_path)
    if only:
        unknown = [i for i in only if i not in suite]
        if unknown:
            raise SystemExit(f"--only: not in suite: {unknown}")
    suite_dir = os.path.dirname(os.path.abspath(suite_path))
    audio = audio_backends()
    jobs = []
    for cid, case in suite.items():
        if only and cid not in only:
            continue
        args = json.loads(json.dumps({k: v for k, v in case.items() if k not in SUITE_ONLY}))
        for k in LISTS:  # allow "refs": ["a.png"] shorthand
            if args.get(k):
                args[k] = [x if isinstance(x, dict) else {"key": x} for x in args[k]]
        if isinstance(args.get("soundtrack"), str):
            args["soundtrack"] = {"key": args["soundtrack"]}
        for box, k in file_slots(args):
            box[k] = resolve(box[k], suite_dir, f"{cid}.{k}")
        args["recipe"] = recipe
        merged = {**(case.get("tuning") or {}), **(tuning or {})}
        if merged:
            args["tuning"] = merged
        else:
            args.pop("tuning", None)
        args["idempotency_key"] = f"bench:{name}:{cid}"
        jobs.append({"name": cid, "tool": "generate_audio" if args.get("backend") in audio else "generate_video", **args})
    return jobs


class Router:
    """batch.py only speaks generate_video / video_url: send audio cases to generate_audio and hand back their audio_url."""

    def __init__(self, tools, audio_keys):
        self.tools, self.audio_keys = tools, audio_keys

    async def __call__(self, tool, **args):
        if tool == "generate_video" and args.get("idempotency_key") in self.audio_keys:
            tool = "generate_audio"
        out = await self.tools(tool, **args)
        if tool == "get_job" and "audio_url" in (out.get("output") or {}):
            out["output"]["video_url"] = out["output"]["audio_url"]
        return out


async def upload_all(jobs, out_dir):
    """Upload every local file the jobs reference (batch.py would only do first/last frame), once per path."""
    done = media.load_json(os.path.join(out_dir, "results.json"), {})
    todo = [j for j in jobs if done.get(j["name"], {}).get("status") not in DONE]
    local = sorted({box[k] for j in todo for box, k in file_slots(j) if not box[k].startswith("inputs/")})
    if not local:
        return
    async with AsyncExitStack() as stack:
        tools = await batch.session_tools(stack)
        keys = {}
        for p in local:
            keys[p] = await upload_file(tools, p)
            print(f"uploaded {os.path.relpath(p)} -> {keys[p]}", flush=True)
    for j in todo:
        for box, k in file_slots(j):
            box[k] = keys.get(box[k], box[k])


async def passthrough_upload(tools, value):
    """batch.py calls upload() on first/last frame; ours are already keys by then."""
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite")
    ap.add_argument("--recipe", required=True, help='render-graph recipe, e.g. "r1"')
    ap.add_argument("--name", required=True, help="run name: output dir and idempotency-key prefix")
    ap.add_argument("--tuning", type=json.loads, help="JSON object of knob overrides")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--out", default=os.path.join("out", "bench"), help="parent dir for runs")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit (no network)")
    a = ap.parse_args()
    if a.tuning is not None and not isinstance(a.tuning, dict):
        ap.error("--tuning must be a JSON object")

    jobs = build_jobs(a.suite, a.recipe, a.tuning, a.name, a.only.split(",") if a.only else None)
    out_dir = os.path.join(a.out, a.name)
    if a.dry_run:
        print(json.dumps(jobs, indent=1))
        n_up = len({box[k] for j in jobs for box, k in file_slots(j) if not box[k].startswith("inputs/")})
        print(f"{len(jobs)} jobs -> {out_dir}/ ({n_up} local files to upload)", file=sys.stderr)
        return

    os.makedirs(out_dir, exist_ok=True)
    asyncio.run(upload_all(jobs, out_dir))
    audio_keys = {j["idempotency_key"] for j in jobs if j["tool"] == "generate_audio"}
    jobs_path = os.path.join(out_dir, "jobs.json")
    json.dump([{k: v for k, v in j.items() if k != "tool"} for j in jobs], open(jobs_path, "w"), indent=1)

    session_tools = batch.session_tools
    batch.upload = passthrough_upload

    async def routed_session_tools(stack):
        return Router(await session_tools(stack), audio_keys)
    batch.session_tools = routed_session_tools
    sys.argv = [batch.__file__, jobs_path, out_dir]
    asyncio.run(batch.main())

    for j in jobs:  # batch.py names every download .mp4
        mp4 = os.path.join(out_dir, j["name"] + ".mp4")
        if j["tool"] == "generate_audio" and os.path.exists(mp4):
            os.replace(mp4, os.path.join(out_dir, j["name"] + ".wav"))


if __name__ == "__main__":
    main()
