"""Submit a batch of jobs through the MCP gateway, wait for all of them, download the results.

  python examples/batch.py jobs.json out/            # jobs.json: [{"name": ..., "prompt": ..., ...}, ...]

Each job takes the generate_video arguments plus two local-only keys:
  name          output file stem (required)
  first_frame / last_frame   LOCAL image paths (uploaded for you)
Writes out/<name>.mp4 and out/results.json (status, timings, cost per job). Re-running skips finished files.
"""
import asyncio
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcp_client import Tools, connect, upload  # noqa: E402
from mcp import ClientSession  # noqa: E402

FINAL = ("SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED", "REJECTED")


async def session_tools(stack):
    read, write = await stack.enter_async_context(connect())
    s = await stack.enter_async_context(ClientSession(read, write))
    await s.initialize()
    return Tools(s, [t.name for t in (await s.list_tools()).tools])


async def main():
    from contextlib import AsyncExitStack
    jobs = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rpath = os.path.join(out, "results.json")
    results = json.load(open(rpath)) if os.path.exists(rpath) else {}

    async with AsyncExitStack() as stack:
        tools = await session_tools(stack)
        keys = {}
        for j in jobs:
            if results.get(j["name"], {}).get("status") in ("SUCCEEDED", "QUEUED", "STARTING", "RUNNING"):
                continue
            args = {k: v for k, v in j.items() if k != "name"}
            for f in ("first_frame", "last_frame"):
                if args.get(f):
                    keys.setdefault(args[f], await upload(tools, args[f]))
                    args[f] = keys[args[f]]
            args.setdefault("idempotency_key", f"batch:{os.path.basename(sys.argv[1])}:{j['name']}")
            try:
                r = await tools("generate_video", **args)
            except SystemExit as exc:  # blocked prompt or bad input: record it and keep going
                results[j["name"]] = {"job_id": None, "status": "REJECTED", "error": str(exc)}
                print(f"rejected {j['name']}: {exc}", flush=True)
                continue
            results[j["name"]] = {"job_id": r["job_id"], "status": r["status"], "submitted": time.time()}
            print(f"queued {j['name']} -> {r['job_id']} (eta {r.get('eta_s')} s)", flush=True)
        json.dump(results, open(rpath, "w"), indent=1)

    t0 = time.time()
    while True:
        pending = [n for n, r in results.items() if r["status"] not in FINAL]
        if not pending:
            break
        async with AsyncExitStack() as stack:
            tools = await session_tools(stack)
            for n in pending:
                st = await tools("get_job", job_id=results[n]["job_id"])
                if st["status"] != results[n]["status"]:
                    print(f"[{time.time() - t0:5.0f}s] {n}: {st['status']}", flush=True)
                results[n].update(status=st["status"], timings=st.get("timings"), cost=st.get("cost"),
                                  error=st.get("error"), finished_at=st.get("finished_at"))
                if st["status"] == "SUCCEEDED":
                    urllib.request.urlretrieve(st["output"]["video_url"], os.path.join(out, f"{n}.mp4"))
        json.dump(results, open(rpath, "w"), indent=1)
        await asyncio.sleep(30)
    usd = sum((r.get("cost") or {}).get("usd") or 0 for r in results.values())
    ok = sum(r["status"] == "SUCCEEDED" for r in results.values())
    print(f"done: {ok}/{len(results)} succeeded, GPU cost USD {usd:.3f}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    asyncio.run(main())
