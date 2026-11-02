"""Minimal MCP client for an open-media-mcp gateway (AWS_IAM / SigV4). Also the end-to-end smoke test.

  pip install mcp httpx botocore
  export GATEWAY_URL=$(aws cloudformation describe-stacks --stack-name OpenMediaMcp \
      --query "Stacks[0].Outputs[?OutputKey=='GatewayUrl'].OutputValue" --output text)
  python examples/mcp_client.py tools                                 # list tools (auth check, no GPU)
  python examples/mcp_client.py call get_capabilities
  python examples/mcp_client.py generate --prompt "..." --out clip.mp4 [--first-frame still.png] [--soundtrack vo.wav]
                                          [--backend wan2.2|h3] [--seconds 5] [--aspect 16:9] [--quality full|draft4]
JWT gateways: set GATEWAY_TOKEN=<access token> instead of using AWS credentials.
"""
import argparse
import asyncio
import json
import mimetypes
import os
import sys
import time
import urllib.request

from contextlib import asynccontextmanager

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session
from mcp import ClientSession

try:  # mcp >= 2: transport takes a pre-built httpx2 client
    import httpx2 as httpx
    from mcp.client.streamable_http import streamable_http_client as _transport
    MCP2 = True
except ImportError:  # mcp 1.x
    import httpx
    from mcp.client.streamable_http import streamablehttp_client as _transport
    MCP2 = False


class SigV4(httpx.Auth):
    """Signs every MCP request for the bedrock-agentcore service with whatever AWS credentials botocore finds."""

    requires_request_body = True

    def __init__(self, region):
        self.creds = Session().get_credentials()
        if self.creds is None:
            sys.exit("no AWS credentials (set AWS_PROFILE or run inside a role)")
        self.region = region

    def auth_flow(self, request):
        req = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=dict(request.headers))
        SigV4Auth(self.creds.get_frozen_credentials(), "bedrock-agentcore", self.region).add_auth(req)
        request.headers.update(dict(req.headers))
        yield request


@asynccontextmanager
async def connect():
    url = os.environ.get("GATEWAY_URL") or sys.exit("set GATEWAY_URL")
    token = os.environ.get("GATEWAY_TOKEN")
    auth = None if token else SigV4(url.split(".bedrock-agentcore.")[1].split(".")[0])
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if MCP2:
        async with httpx.AsyncClient(auth=auth, headers=headers, timeout=httpx.Timeout(60, read=300)) as client:
            async with _transport(url, http_client=client) as streams:
                yield streams[0], streams[1]
    else:
        async with _transport(url, auth=auth, headers=headers, timeout=60) as streams:
            yield streams[0], streams[1]


def payload(result):
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text, "isError": result.isError}


class Tools:
    def __init__(self, session, names):
        self.s = session
        self.prefix = {n.split("___")[-1]: n for n in names}

    async def __call__(self, tool, **args):
        out = payload(await self.s.call_tool(self.prefix[tool], args))
        if out.get("ok") is False or "raw" in out:
            raise SystemExit(f"{tool} failed: {json.dumps(out)[:1500]}")
        return out


async def upload(tools, path):
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    up = await tools("create_upload", filename=os.path.basename(path), content_type=ctype)
    with open(path, "rb") as fh:
        req = urllib.request.Request(up["upload_url"], data=fh.read(), method="PUT", headers=up["headers"])
    urllib.request.urlopen(req, timeout=300).read()
    return up["key"]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tools", "call", "generate"])
    ap.add_argument("tool", nargs="?")
    ap.add_argument("--args", default="{}", help="JSON arguments for `call`")
    ap.add_argument("--prompt")
    ap.add_argument("--backend")
    ap.add_argument("--first-frame")
    ap.add_argument("--last-frame")
    ap.add_argument("--soundtrack")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--aspect")
    ap.add_argument("--quality")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="clip.mp4")
    a = ap.parse_args()

    async with connect() as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = await session.list_tools()
        names = [t.name for t in listed.tools]
        if a.cmd == "tools":
            for t in listed.tools:
                print(f"{t.name}: {t.description[:110]}")
            return
        tools = Tools(session, names)
        if a.cmd == "call":
            print(json.dumps(await tools(a.tool, **json.loads(a.args)), indent=1))
            return

        req = {k: v for k, v in dict(prompt=a.prompt, backend=a.backend, seconds=a.seconds, aspect=a.aspect,
                                     quality=a.quality, seed=a.seed).items() if v is not None}
        if a.first_frame:
            req["first_frame"] = await upload(tools, a.first_frame)
        if a.last_frame:
            req["last_frame"] = await upload(tools, a.last_frame)
        if a.soundtrack:
            req["soundtrack"] = {"key": await upload(tools, a.soundtrack)}
        job = await tools("generate_video", **req)
        print(json.dumps(job, indent=1), flush=True)
        t0, last = time.time(), None
        while True:
            st = await tools("get_job", job_id=job["job_id"])
            line = f"{st['status']}: {st.get('phase', '')}"
            if line != last:
                print(f"[{time.time() - t0:5.0f}s] {line}", flush=True)
                last = line
            if st["status"] in ("SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"):
                break
            await asyncio.sleep(20)
        print(json.dumps(st, indent=1))
        if st["status"] == "SUCCEEDED":
            urllib.request.urlretrieve(st["output"]["video_url"], a.out)
            print(f"saved {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
