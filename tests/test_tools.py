import importlib.util
import json
import os
import sys

import pytest

os.environ.update(AWS_DEFAULT_REGION="us-east-1", AWS_REGION="us-east-1", MEDIA_TABLE="t", MEDIA_QUEUE_URL="q", MEDIA_BUCKET="b",
                  MEDIA_ASG="a", MEDIA_BACKENDS="wan2.2,h3,stable-audio-3,qwen-image,h3-image", MEDIA_MAX_QUEUED="50", MEDIA_JOB_RETENTION_DAYS="90")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))  # identity.py, bundled next to handler.py
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


def test_image_defaults_and_routing():
    s = tools.normalize_image({"prompt": "a lighthouse at dusk"}, CALLER)
    assert (s["backend"], s["kind"], s["aspect"], s["steps"], s["cfg"], s["count"]) == ("qwen-image", "image", "16:9", 30, 3.0, 1)
    s = tools.normalize_image({"prompt": "x", "aspect": "3:4", "steps": 20, "cfg": 4.5, "count": 4, "seed": 9,
                               "negative_prompt": "text"}, CALLER)
    assert (s["aspect"], s["steps"], s["cfg"], s["count"], s["seed"], s["negative_prompt"]) == ("3:4", 20, 4.5, 4, 9, "text")
    with pytest.raises(tools.ToolError):
        tools.normalize({"prompt": "x", "backend": "qwen-image"}, CALLER)
    with pytest.raises(tools.ToolError):
        tools.normalize_audio({"prompt": "x", "backend": "qwen-image"}, CALLER)
    assert tools.estimate("qwen-image", None, "full", 2, 30) == int(4.0 + 0.85 * 60)


def test_image_limits():
    for bad in ({"count": 5}, {"count": 2.5}, {"steps": 100}, {"cfg": 20}, {"aspect": "21:9"}, {"backend": "wan2.2"}, {"prompt": ""}):
        with pytest.raises(tools.ToolError):
            tools.normalize_image({"prompt": "x", **bad}, CALLER)


def test_get_job_returns_image_urls(monkeypatch):
    out = {"images": [{"key": f"outputs/j-1/j-1_{i}.png", "seed": 6 + i, "width": 1664, "height": 928, "sha256": f"h{i}"}
                      for i in (1, 2)], "manifest_key": "outputs/j-1/manifest.json"}
    item = {"job_id": "j-1", "status": "SUCCEEDED", "backend": "qwen-image", "created_at": "2026-11-09T00:00:00+00:00",
            "tenant": CALLER["tenant"], "output": out,
            "spec": json.dumps({"backend": "qwen-image", "kind": "image", "aspect": "16:9", "count": 2, "steps": 30, "cfg": 3.0, "seed": 7})}
    monkeypatch.setattr(tools, "_own", lambda job_id, caller: item)
    monkeypatch.setattr(tools, "_url", lambda key: "https://signed/" + key)
    r = tools.get_job({"job_id": "j-1"}, CALLER)
    assert r["output"]["image_url"] == "https://signed/outputs/j-1/j-1_1.png" and len(r["output"]["images"]) == 2
    assert r["output"]["images"][1]["seed"] == 8 and r["request"]["count"] == 2 and "notice" not in r


def test_tools_json_matches_the_lambda():
    with open(os.path.join(os.path.dirname(__file__), "..", "tools.json")) as fh:
        assert {t["name"] for t in json.load(fh)} == set(tools.TOOLS)


def test_wan_rejects_references():
    with pytest.raises(tools.ToolError):
        tools.normalize({"prompt": "x", "refs": [{"key": "k"}]}, CALLER)


def test_recipes_resolve_and_tuning_is_validated():
    s = tools.normalize({"prompt": "x"}, CALLER)
    assert s["recipe"] == "r0" and s["tuning"]["interpolate"] == 1
    s = tools.normalize({"prompt": "x", "recipe": "r1", "tuning": {"full_sampler": "official", "full_steps": 30}}, CALLER)
    assert s["tuning"]["interpolate"] == 2 and s["tuning"]["full_steps"] == 30 and s["tuning"]["loras"] == "2509"
    for bad in ({"recipe": "r9"}, {"tuning": {"nope": 1}}, {"tuning": {"interpolate": 3}}, {"tuning": {"full_steps": 500}}):
        with pytest.raises(tools.ToolError):
            tools.normalize({"prompt": "x", **bad}, CALLER)
    h = tools.normalize({"prompt": "x", "backend": "h3", "recipe": "r2"}, CALLER)
    assert h["tuning"]["enhance_prompt"] is True and h["tuning"]["turbo"] == "768p"


def test_h3_image_normalize(monkeypatch):
    monkeypatch.setattr(tools.s3, "head_object", lambda **kw: {})
    key = f"inputs/{tools.tenant_prefix(CALLER['tenant'])}/a.png"
    s = tools.normalize_image({"prompt": "Replace the taxi in <Picture 1> with a pink ice-cream truck", "backend": "h3-image",
                               "refs": [{"key": key}], "quality": "draft"}, CALLER)
    assert (s["aspect"], s["steps"], s["cfg"], s["quality"], s["refs"]) == ("auto", 6, 1.0, "draft", [{"key": key}])
    assert s["tuning"] == {"enhance_prompt": True, "easyfix": "on"} and s["recipe"] == "r0"
    s = tools.normalize_image({"prompt": "x", "backend": "h3-image", "tuning": {"enhance_prompt": False}}, CALLER)
    assert s["tuning"]["enhance_prompt"] is False and s["quality"] == "full" and s["refs"] == []
    for mode in ("on", "off", "compare"):
        assert tools.normalize_image({"prompt": "x", "backend": "h3-image", "tuning": {"easyfix": mode}}, CALLER)["tuning"]["easyfix"] == mode
    with pytest.raises(tools.ToolError):
        tools.normalize_image({"prompt": "x", "backend": "h3-image", "tuning": {"easyfix": "both"}}, CALLER)
    for bad in ({"steps": 20}, {"cfg": 3}, {"negative_prompt": "blur"}, {"refs": [{"key": key}] * 17}, {"quality": "draft8"},
                {"aspect": "7:5"}):
        with pytest.raises(tools.ToolError):
            tools.normalize_image({"prompt": "x", "backend": "h3-image", **bad}, CALLER)
    with pytest.raises(tools.ToolError):
        tools.normalize_image({"prompt": "x", "refs": [{"key": key}]}, CALLER)  # qwen-image takes no refs
    assert tools.normalize_image({"prompt": "x"}, CALLER)["backend"] == "qwen-image"


def test_h3_image_takes_a_full_caption_and_estimates_its_six_steps():
    assert tools.normalize_image({"prompt": "x" * 9000, "backend": "h3-image"}, CALLER)["prompt"] == "x" * 9000
    with pytest.raises(tools.ToolError):
        tools.normalize_image({"prompt": "x" * 9000}, CALLER)
    assert tools.estimate("h3-image", None, "full", 2, 6) == int(20.0 + 4.0 * 12)


def test_short_jobs_share_gpus_and_long_jobs_get_their_own(monkeypatch):
    monkeypatch.setenv("MEDIA_MAX_INSTANCES", "14")
    f = lambda queued, in_flight=0: {"queued": queued, "in_flight": in_flight}
    assert tools.gpus_wanted(f(0)) == 0
    assert tools.gpus_wanted(f(9), 44) == 2          # nine 44 s images: two GPUs, not nine
    assert tools.gpus_wanted(f(1), 44) == 1
    assert tools.gpus_wanted(f(1, 1), 44) == 2       # the only GPU is busy (maybe on a long video): add one
    assert tools.gpus_wanted(f(3), 600) == 3         # videos: one each
    assert tools.gpus_wanted(f(80), 600) == 14       # capped


def test_reconcile_only_starts_the_first_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(tools.asg, "set_desired_capacity", lambda **kw: calls.append(kw["DesiredCapacity"]))
    fleet = {"queued": 5, "in_flight": 0, "desired": 0, "instances": []}
    monkeypatch.setattr(tools, "fleet", lambda: fleet)
    tools.reconcile()
    fleet.update(desired=2, instances=[{"state": "InService"}])
    tools.reconcile()
    assert calls == [1]
