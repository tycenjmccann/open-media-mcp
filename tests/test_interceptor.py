import base64
import importlib.util
import json
import os
from unittest import mock

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
spec = importlib.util.spec_from_file_location("icpt", os.path.join(os.path.dirname(__file__), "..", "lambda", "interceptor", "handler.py"))
icpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icpt)


def call(headers, args):
    return {"mcp": {"gatewayRequest": {"headers": headers, "body": {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                                     "params": {"name": "media___get_job", "arguments": args}}}}}


def test_jwt_caller_is_stamped_and_spoofing_is_dropped():
    claims = {"iss": "https://idp", "sub": "u-123", "email": "a@b.co"}
    tok = "x." + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=") + ".y"
    out = icpt.lambda_handler(call({"Authorization": f"Bearer {tok}"}, {"job_id": "j-1", "_caller": {"tenant": "evil"}}), None)
    c = out["mcp"]["transformedGatewayRequest"]["body"]["params"]["arguments"]["_caller"]
    assert c["tenant"] == "jwt:https://idp|u-123" and c["principal"] == "a@b.co" and c["auth"] == "jwt"


def test_sigv4_caller_maps_to_account():
    with mock.patch.object(icpt.sts, "get_access_key_info", return_value={"Account": "111122223333"}):
        h = {"authorization": "AWS4-HMAC-SHA256 Credential=ASIAEXAMPLEKEY12345/20261102/us-east-1/bedrock-agentcore/aws4_request, SignedHeaders=host, Signature=x"}
        out = icpt.lambda_handler(call(h, {}), None)
    c = out["mcp"]["transformedGatewayRequest"]["body"]["params"]["arguments"]["_caller"]
    assert c["tenant"] == "aws:111122223333" and c["auth"] == "sigv4"


def test_non_tool_calls_pass_through():
    ev = {"mcp": {"gatewayRequest": {"body": {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}}}}
    assert icpt.lambda_handler(ev, None)["mcp"]["transformedGatewayRequest"]["body"] == {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
