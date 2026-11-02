import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
from backends import load  # noqa: E402

B = load(["wan2.2", "h3"])


def prep(guides=(), soundtrack=None):
    return {"refs": [], "ref_videos": [], "ref_audios": [], "guides": list(guides), "soundtrack": soundtrack}


def test_frame_cadence():
    assert B["wan2.2"].frames(5) == 81 and (B["wan2.2"].frames(3.3) - 1) % 4 == 0
    assert B["h3"].frames(5.17) == 124 and B["h3"].frames(6) % 17 == 5


def test_wan_t2v_uses_t2v_experts_and_split_steps():
    g = B["wan2.2"].build_graph({"prompt": "a lighthouse at dusk", "aspect": "16:9", "seconds": 5, "quality": "full", "seed": 1},
                                prep(), {}, "j1")
    assert "t2v_high" in g["unet_high"]["inputs"]["unet_name"] and "t2v_low" in g["unet_low"]["inputs"]["unet_name"]
    assert g["latent"]["inputs"] == {"width": 1280, "height": 720, "length": 81, "batch_size": 1}
    hi, lo = g["sample_high"]["inputs"], g["sample_low"]["inputs"]
    assert (hi["start_at_step"], hi["end_at_step"], lo["start_at_step"]) == (0, 10, 10)
    assert lo["latent_image"] == ["sample_high", 0] and "lora_high" not in g


def test_wan_i2v_and_flf_and_draft_loras():
    spec = {"prompt": "p", "aspect": "9:16", "seconds": 5, "quality": "draft4", "seed": 1, "first_frame": "k"}
    g = B["wan2.2"].build_graph(spec, prep([("image", "a.png", 0.0)]), {"a.png": "A"}, "j2")
    assert g["cond"]["class_type"] == "WanImageToVideo" and g["first"]["inputs"]["image"] == "A"
    assert "i2v" in g["lora_high"]["inputs"]["lora_name"] and g["sample_high"]["inputs"]["cfg"] == 1.0
    g = B["wan2.2"].build_graph(spec, prep([("image", "a.png", 0.0), ("image", "b.png", "end")]), {"a.png": "A", "b.png": "B"}, "j3")
    assert g["cond"]["class_type"] == "WanFirstLastFrameToVideo" and g["cond"]["inputs"]["end_image"] == ["last", 0]


def test_h3_soundtrack_guides_conditioning_only():
    spec = {"prompt": "<Picture 1> sings", "aspect": "16:9", "seconds": 5.17, "quality": "full", "seed": 3,
            "soundtrack": {"at": 0.0}}
    g = B["h3"].build_graph(spec, prep([("image", "f.png", 0.0)], ("s.wav", 5.17)), {"f.png": "F", "s.wav": "S"}, "j4")
    assert g["guider"]["inputs"]["conditioning"] == ["track_guide", 0]
    assert g["sample"]["inputs"]["latent_image"] == ["r2v", 1]
    assert g["sched"]["inputs"]["steps"] == 20


def test_licenses_and_notices():
    assert B["wan2.2"].license.startswith("Apache-2.0") and not B["wan2.2"].notice
    assert B["h3"].notice == "Powered by MiniMax H3"
