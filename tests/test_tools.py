import importlib.util
import os

import pytest

os.environ.update(AWS_DEFAULT_REGION="us-east-1", AWS_REGION="us-east-1", MEDIA_TABLE="t", MEDIA_QUEUE_URL="q", MEDIA_BUCKET="b",
                  MEDIA_ASG="a", MEDIA_BACKENDS="wan2.2,h3,stable-audio-3", MEDIA_MAX_QUEUED="50", MEDIA_JOB_RETENTION_DAYS="90")
spec = importlib.util.spec_from_file_location("tools", os.path.join(os.path.dirname(__file__), "..", "lambda", "tools", "handler.py"))
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
CALLER = {"tenant": "aws:111122223333", "principal": "p", "auth": "sigv4"}


def test_default_video_backend_is_wan_and_audio_backends_are_routed():
    assert tools.DEFAULT_BACKEND == "wan2.2"
    with pytest.raises(tools.ToolError):
        tools.normalize({"prompt": "x", "backend": "stable-audio-3"}, CALLER)
    a = tools.normalize_audio({"prompt": "lo-fi beat, 82 BPM", "seconds": 45}, CALLER)
    assert a["backend"] == "stable-audio-3" and a["kind"] == "audio" and a["model"] == "medium" and a["seconds"] == 45


def test_audio_limits():
    with pytest.raises(tools.ToolError):
        tools.normalize_audio({"prompt": "x", "seconds": 999}, CALLER)
    with pytest.raises(tools.ToolError):
        tools.normalize_audio({"prompt": "x", "model": "large"}, CALLER)


def test_wan_rejects_references():
    with pytest.raises(tools.ToolError):
        tools.normalize({"prompt": "x", "refs": [{"key": "k"}]}, CALLER)
