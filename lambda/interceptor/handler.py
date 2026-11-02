"""AgentCore Gateway REQUEST interceptor: stamp the verified caller onto every tools/call.

The gateway has already authenticated the request (SigV4 or JWT) before this runs. We read who it was from the
forwarded Authorization header and add `_caller` = {tenant, principal, auth} to the tool arguments, replacing any
`_caller` a client tried to send. The tools Lambda scopes jobs, uploads and quotas to `tenant`.

  JWT  (CUSTOM_JWT gateways): tenant = the MEDIA_TENANT_CLAIM claim (default "sub"); principal = email/username/client_id.
  SigV4 (AWS_IAM gateways):   tenant = "aws:<account id>" of the signing access key; principal adds the key id.
"""
import base64
import json
import os
import re

import boto3

sts = boto3.client("sts")
_ACCOUNTS = {}
CLAIM = os.environ.get("MEDIA_TENANT_CLAIM", "sub")


def _jwt_claims(token):
    # Signature already verified by the gateway authorizer; we only read the claims.
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def identify(headers):
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        c = _jwt_claims(auth.split(None, 1)[1])
        principal = c.get("email") or c.get("username") or c.get("cognito:username") or c.get("client_id") or c.get("sub")
        return {"tenant": f"jwt:{c.get('iss', '')}|{c.get(CLAIM, c.get('sub'))}", "principal": principal, "auth": "jwt",
                "groups": c.get("cognito:groups") or c.get("groups") or []}
    m = re.search(r"Credential=([A-Z0-9]{16,128})/", auth)
    if m:
        akid = m.group(1)
        if akid not in _ACCOUNTS:
            _ACCOUNTS[akid] = sts.get_access_key_info(AccessKeyId=akid)["Account"]
        acct = _ACCOUNTS[akid]
        return {"tenant": f"aws:{acct}", "principal": f"aws:{acct}/{akid[:8]}...", "auth": "sigv4"}
    return {"tenant": "unverified", "principal": "unknown", "auth": "unknown"}


def lambda_handler(event, context):
    req = (event.get("mcp") or {}).get("gatewayRequest") or {}
    body = req.get("body") or {}
    if isinstance(body, dict) and body.get("method") == "tools/call":
        headers = {k.lower(): v for k, v in (req.get("headers") or {}).items()}
        params = body.setdefault("params", {})
        args = params.get("arguments") or {}
        args.pop("_caller", None)
        args["_caller"] = identify(headers)
        params["arguments"] = args
        print(json.dumps({"event": "tools_call", "tool": params.get("name"), "tenant": args["_caller"]["tenant"],
                          "auth": args["_caller"]["auth"]}))
    return {"interceptorOutputVersion": "1.0", "mcp": {"transformedGatewayRequest": {"body": body}}}
