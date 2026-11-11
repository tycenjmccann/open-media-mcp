import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
import identity  # noqa: E402

from test_tools import CALLER, tools  # noqa: E402


def face(px=200, occluded=False):
    return {"face_px": px, "occluded": occluded}


def test_verdicts_follow_the_calibrated_rules():
    assert identity.verdict(99.0, face())[0] == "PASS"
    assert identity.verdict(93.0, face())[0] == "REVIEW"
    assert identity.verdict(62.0, face())[0] == "FAIL"
    # a low score on a small or covered face is not a FAIL: the real character scored 21 at 85 px
    assert identity.verdict(21.0, face(px=85))[0] == "REVIEW"
    assert identity.verdict(99.0, face(occluded=True))[0] == "REVIEW"


def frame(t, score, clear=True):
    return {"t": t, "score": score, "clear": clear}


def test_video_fails_on_any_clear_frame_that_does_not_match():
    a = identity.aggregate_frames([frame(0.5, 99), frame(1.5, 98), frame(2.5, 61)])
    assert a["verdict"] == "FAIL" and "2.5s" in a["reason"]
    assert identity.aggregate_frames([frame(0.5, 99), frame(1.5, 98.5)])["verdict"] == "PASS"
    assert identity.aggregate_frames([frame(0.5, 99), frame(1.5, 94)])["verdict"] == "REVIEW"
    assert identity.aggregate_frames([frame(0.5, 20, clear=False)])["verdict"] == "REVIEW"


def test_identity_check_uses_refs_by_default(monkeypatch):
    monkeypatch.setattr(tools, "_key", lambda k, caller, what: k)
    s = tools.normalize({"prompt": "x", "backend": "h3", "refs": [{"key": "sheet"}], "identity_check": {}}, CALLER)
    assert s["identity_check"] == {"reference": ["sheet"], "pass_at": 97.0, "review_at": 90.0, "min_face": 100}
    s = tools.normalize({"prompt": "x", "identity_check": {"reference": [{"key": "a"}], "pass_at": 95}}, CALLER)
    assert s["identity_check"]["reference"] == ["a"] and s["identity_check"]["pass_at"] == 95
    with pytest.raises(tools.ToolError):
        tools.normalize({"prompt": "x", "identity_check": {}}, CALLER)  # nothing to compare against
    with pytest.raises(tools.ToolError):
        tools.normalize({"prompt": "x", "identity_check": {"reference": ["a"], "pass_at": 80, "review_at": 90}}, CALLER)


def test_check_identity_validates_its_inputs(monkeypatch):
    monkeypatch.setattr(tools, "_key", lambda k, caller, what: k)
    with pytest.raises(tools.ToolError):
        tools.check_identity({"reference": [], "image": "x"}, CALLER)
    with pytest.raises(tools.ToolError):
        tools.check_identity({"reference": [{"key": str(i)} for i in range(6)], "image": "x"}, CALLER)
    with pytest.raises(tools.ToolError):
        tools.check_identity({"reference": [{"key": "a"}]}, CALLER)
    assert "check_identity" in tools.TOOLS
