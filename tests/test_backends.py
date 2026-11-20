import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
from backends import load  # noqa: E402

B = load(["wan2.2", "h3", "stable-audio-3", "qwen-image", "h3-image"])


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
    hi, lo = g["sample0"]["inputs"], g["sample1"]["inputs"]
    assert (hi["start_at_step"], hi["end_at_step"], lo["start_at_step"]) == (0, 10, 10)
    assert lo["latent_image"] == ["sample0", 0] and "lora_high" not in g


def test_wan_i2v_and_flf_and_draft_loras():
    spec = {"prompt": "p", "aspect": "9:16", "seconds": 5, "quality": "draft4", "seed": 1, "first_frame": "k"}
    g = B["wan2.2"].build_graph(spec, prep([("image", "a.png", 0.0)]), {"a.png": "A"}, "j2")
    assert g["cond"]["class_type"] == "WanImageToVideo" and g["first"]["inputs"]["image"] == "A"
    assert "i2v" in g["lora_high"]["inputs"]["lora_name"] and g["sample0"]["inputs"]["cfg"] == 1.0
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


def test_stable_audio_graph():
    g = B["stable-audio-3"].build_graph({"prompt": "rain on a tin roof", "seconds": 12, "seed": 5, "model": "small-sfx"}, prep(), {}, "j5")
    assert g["ckpt"]["inputs"]["ckpt_name"] == "stable_audio_3_small_sfx.safetensors"
    assert g["clip"]["inputs"]["type"] == "stable_audio" and g["latent"]["inputs"]["seconds"] == 12.0
    assert g["sample"]["inputs"]["steps"] == 8 and g["decode"]["inputs"]["vae"] == ["ckpt", 2]
    assert B["stable-audio-3"].notice == "Powered by Stability AI"


def test_qwen_image_graph():
    q = B["qwen-image"]
    g = q.build_graph({"prompt": "a lighthouse", "aspect": "9:16", "steps": 30, "cfg": 3.0, "count": 1, "seed": 7}, prep(), {}, "j6")
    assert g["unet"]["inputs"]["unet_name"] == "qwen_image_fp8_e4m3fn.safetensors" and g["shift"]["inputs"]["shift"] == 3.1
    assert g["clip"]["inputs"] == {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}
    assert g["lat"]["inputs"] == {"width": 928, "height": 1664, "batch_size": 1}
    assert g["neg"]["inputs"]["text"] == "blurry, lowres, deformed, watermark"
    ks = g["ks"]["inputs"]
    assert (ks["sampler_name"], ks["scheduler"], ks["cfg"], ks["steps"], ks["seed"]) == ("euler", "simple", 3.0, 30, 7)
    assert ks["model"] == ["shift", 0] and g["save"]["inputs"]["images"] == ["dec", 0] and "ks_2" not in g
    g = q.build_graph({"prompt": "p", "aspect": "1:1", "steps": 20, "cfg": 4.5, "count": 3, "seed": 7,
                       "negative_prompt": "text"}, prep(), {}, "j7")
    assert [g[k]["inputs"]["seed"] for k in ("ks", "ks_2", "ks_3")] == [7, 8, 9] and "ks_4" not in g
    assert g["ks_3"]["inputs"]["steps"] == 20 and g["ks_3"]["inputs"]["cfg"] == 4.5 and g["neg"]["inputs"]["text"] == "text"
    assert g["save_3"]["inputs"]["filename_prefix"] == "jobs/j7_3" and g["dec_2"]["inputs"]["samples"] == ["ks_2", 0]
    assert {s for _, _, s in q.models} == {"diffusion_models", "text_encoders", "vae"}
    assert q.kind == "image" and q.license.startswith("Apache-2.0") and not q.notice


def test_wan_official_full_splits_at_the_expert_boundary():
    t2v = {"prompt": "p", "aspect": "16:9", "seconds": 5, "quality": "full", "seed": 1, "tuning": {"full_sampler": "official"}}
    g = B["wan2.2"].build_graph(t2v, prep(), {}, "j")
    hi, lo = g["sample0"]["inputs"], g["sample1"]["inputs"]
    assert (hi["steps"], hi["end_at_step"], hi["cfg"], lo["cfg"], hi["sampler_name"]) == (40, 26, 4.0, 3.0, "uni_pc")
    assert g["shift_high"]["inputs"]["shift"] == 12.0
    i2v = {**t2v, "first_frame": "k"}
    g = B["wan2.2"].build_graph(i2v, prep([("image", "a.png", 0.0)]), {"a.png": "A"}, "j")
    assert g["sample0"]["inputs"]["end_at_step"] == 15 and g["shift_low"]["inputs"]["shift"] == 5.0


def test_wan_balanced_runs_base_steps_before_the_lora():
    spec = {"prompt": "p", "aspect": "16:9", "seconds": 5, "quality": "balanced", "seed": 1,
            "tuning": {"shift_draft": 5, "loras": "2509", "interpolate": 2, "encode": "hq"}}
    g = B["wan2.2"].build_graph(spec, prep(), {}, "j")
    s0, s1, s2 = (g[f"sample{i}"]["inputs"] for i in range(3))
    assert s0["model"] == ["shift_high", 0] and s0["cfg"] == 3.5 and (s0["start_at_step"], s0["end_at_step"]) == (0, 2)
    assert s1["model"] == ["shift_high_lora", 0] and s2["model"] == ["shift_low_lora", 0] and s1["end_at_step"] == 4
    assert "250928" in g["lora_high"]["inputs"]["lora_name"]
    assert g["interp"]["inputs"]["multiplier"] == 2 and g["video"]["inputs"]["fps"] == 32
    assert g["save"]["inputs"]["format"] == "mov"


def test_h3_r1_knobs():
    spec = {"prompt": "<Picture 2> waves", "aspect": "1:1", "seconds": 5, "quality": "draft8", "seed": 1,
            "tuning": {"turbo": "768p", "canvas": "official", "keyframe_refs": True}}
    g = B["h3"].build_graph(spec, {**prep([("image", "f.png", 0.0)]), "refs": ["r.png"]}, {"f.png": "F", "r.png": "R"}, "j")
    assert (g["r2v"]["inputs"]["width"], g["r2v"]["inputs"]["height"]) == (768, 768)
    assert g["shift"]["inputs"]["shift_video"] == 6.0 and g["guider"]["inputs"]["model"] == ["shift", 0]
    assert g["r2v"]["inputs"]["ref_images.ref_image_1"] == ["ref1", 0] and g["ref1"]["inputs"]["image"] == "F"
    full = {**spec, "quality": "full", "tuning": {"steps": 50, "sampler": "euler", "regen_2k": True}}
    g = B["h3"].build_graph(full, prep(), {}, "j")
    assert g["sched"]["inputs"]["steps"] == 50 and g["decode"]["inputs"]["samples"] == ["regen", 0]
    assert g["decode_audio"]["inputs"]["samples"] == ["sample", 0]


def test_h3_image_t2i_graph():
    m = B["h3-image"]
    off = {"tuning": {"easyfix": "off"}}  # the stock template sampler
    g = m.build_graph({"prompt": "a lighthouse", "aspect": "16:9", "quality": "full", "count": 1, "seed": 7, **off}, prep(), {}, "j8")
    assert g["unet"]["inputs"]["unet_name"] == "minimax_h3_image_pruned_int8_convrot.safetensors"
    assert g["clip"]["inputs"] == {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}
    c = g["cond"]["inputs"]
    assert (c["width"], c["height"]) == (5472, 3072) and not any(k.startswith("images.") for k in c)
    ks = g["ks"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (6, 1.0, "euler", "simple")
    assert ks["latent_image"] == ["cond", 1] and ks["negative"] == ["neg", 0] and "ks_2" not in g
    assert "steer" not in g and "renoise" not in g
    g = m.build_graph({"prompt": "p", "aspect": "auto", "quality": "draft", "count": 2, "seed": 7, **off}, prep(), {}, "j9")
    assert (g["cond"]["inputs"]["width"], g["cond"]["inputs"]["height"]) == (2048, 2048)
    assert [g[k]["inputs"]["seed"] for k in ("ks", "ks_2")] == [7, 8] and g["save_2"]["inputs"]["filename_prefix"] == "jobs/j9_2"
    assert m.kind == "image" and m.notice == "Powered by MiniMax-H3-Image"
    # the text encoder is the h3 video backend's file, so enabling both fetches it once
    assert set(m.models) & set(B["h3"].models)


def test_h3_image_easyfix_is_on_by_default_and_compare_draws_each_seed_both_ways(tmp_path):
    m = B["h3-image"]
    spec = {"prompt": "p", "aspect": "1:1", "quality": "draft", "count": 1, "seed": 7}
    g = m.build_graph(spec, prep(), {}, "j")  # default: EasyFix, the pack's own workflow settings
    assert g["ks"]["class_type"] == "SamplerCustomAdvanced" and "neg" not in g
    assert g["ks"]["inputs"] == {"noise": ["noise", 0], "guider": ["guide", 0], "sampler": ["renoise", 0],
                                 "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}
    assert g["noise"]["inputs"]["noise_seed"] == 7 and g["guide"]["inputs"]["model"] == ["steer", 0]
    assert g["steer"]["inputs"] == {"model": ["unet", 0], "strength": 0.7, "steps": "0"}
    assert g["renoise"]["inputs"] == {"plan": "RNmid", "custom_plan": "SSRRRS", "s_noise": 0.95, "sampler": ["inner", 0]}
    assert g["inner"]["inputs"]["sampler_name"] == "euler_ancestral" and g["sigmas"]["inputs"]["sigmas"].startswith("1.0, 0.9675")
    assert m.image_meta(spec) == [{"seed": 7, "easyfix": "on"}]
    both = {**spec, "count": 2, "tuning": {"easyfix": "compare"}}
    g = m.build_graph(both, prep(), {}, "j")  # EasyFix then stock for each seed, in the order the worker names the files
    assert [g[k]["class_type"] for k in ("ks", "ks_2", "ks_3", "ks_4")] == ["SamplerCustomAdvanced", "KSampler"] * 2
    assert (g["noise"]["inputs"]["noise_seed"], g["ks_2"]["inputs"]["seed"], g["noise_3"]["inputs"]["noise_seed"],
            g["ks_4"]["inputs"]["seed"]) == (7, 7, 8, 8)
    assert [g[k]["inputs"]["filename_prefix"] for k in ("save", "save_2", "save_3", "save_4")] == [f"jobs/j_{i}" for i in (1, 2, 3, 4)]
    assert m.image_meta(both) == [{"seed": 7, "easyfix": "on"}, {"seed": 7, "easyfix": "off"},
                                  {"seed": 8, "easyfix": "on"}, {"seed": 8, "easyfix": "off"}]
    from PIL import Image  # an edit always samples the stock way, whatever the knob says
    a = str(tmp_path / "a.png")
    Image.new("RGB", (800, 800)).save(a)
    g = m.build_graph({**both, "count": 1}, {**prep(), "refs": [a]}, {a: "A"}, "j")
    assert g["ks"]["class_type"] == "KSampler" and "ks_2" not in g and "steer" not in g
    assert m.image_meta({**both, "count": 1, "refs": [{"key": "k"}]}) == [{"seed": 7, "easyfix": "off"}]


def test_h3_image_edit_wires_refs_in_order_and_matches_the_edited_ratio(tmp_path):
    from PIL import Image
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    Image.new("RGB", (1200, 800)).save(a)
    Image.new("RGB", (500, 500)).save(b)
    g = B["h3-image"].build_graph({"prompt": "Put <Picture 2> on <Picture 1>", "aspect": "auto", "seed": 1},
                                  {**prep(), "refs": [a, b]}, {a: "A", b: "B"}, "j10")
    c = g["cond"]["inputs"]
    assert c["images.image_1"] == ["ref1", 0] and g["ref1"]["inputs"]["image"] == "A" and g["ref2"]["inputs"]["image"] == "B"
    assert (c["width"], c["height"]) == (5024, 3360)  # 3:2 like <Picture 1>, MiniMax's 16 MP size
    assert B["h3-image"].canvas({"aspect": "4:5"}, prep()) == (3648, 4576)
