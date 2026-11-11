"""Wan 2.2 A14B (Wan-AI, Apache-2.0): text-to-video and image-to-video, 720p, 16 fps, silent output.

Two-expert MoE: a high-noise model denoises the early (high-sigma) part of the schedule, a low-noise model the rest
(two KSamplerAdvanced passes). Draft qualities add the lightx2v 4-step distillation LoRAs (cfg 1); "balanced" runs a
couple of full-strength base steps first, which restores the motion the LoRAs damp. A soundtrack is muxed onto the
silent render (no lip-sync on this backend).

Every knob comes from spec["tuning"] (resolved from a named recipe by the tools Lambda). Missing knobs fall back to
recipe r0, the original graph, so jobs queued before a recipe existed render exactly as they used to.
"""

from . import finish

FPS = 16
SIZES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960), "4:5": (768, 960)}
REPO = "Comfy-Org/Wan-2.2"
UNET = {
    ("t2v", "high"): "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    ("t2v", "low"): "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    ("i2v", "high"): "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    ("i2v", "low"): "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
}
LORAS = {
    "v1": {
        ("t2v", "high"): "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        ("t2v", "low"): "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
        ("i2v", "high"): "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        ("i2v", "low"): "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    },
    # lightx2v's late-2025 releases: better camera control and motion than v1.1 / v1 (their release notes).
    "2509": {
        ("t2v", "high"): "Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
        ("t2v", "low"): "Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
        ("i2v", "high"): "Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors",
        ("i2v", "low"): "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
    },
}
LORA_SOURCES = {
    LORAS["2509"][("t2v", "high")]: ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/"),
    LORAS["2509"][("t2v", "low")]: ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/"),
    LORAS["2509"][("i2v", "high")]: ("Kijai/WanVideo_comfy", "LoRAs/Wan22_Lightx2v/"),
    LORAS["2509"][("i2v", "low")]: ("lightx2v/Wan2.2-Distill-Loras", ""),
}
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
RIFE = "rife_v4.26.safetensors"
# quality -> (steps, cfg, use the 4-step LoRAs) for recipe r0
QUALITY = {"full": (20, 3.5, False), "draft8": (8, 1.0, True), "draft4": (4, 1.0, True)}
# Wan-Video/Wan2.2 wan/configs: sample_shift, guide_scale (low expert, high expert), boundary sigma, sample_steps
OFFICIAL = {"t2v": {"shift": 12.0, "cfg_high": 4.0, "cfg_low": 3.0, "boundary": 0.875, "steps": 40},
            "i2v": {"shift": 5.0, "cfg_high": 3.5, "cfg_low": 3.5, "boundary": 0.900, "steps": 40}}
R0 = {"shift_draft": 8, "split": "half", "loras": "v1", "full_sampler": "r0", "full_steps": 0, "interpolate": 1,
      "upscale": "none", "upscale_height": 1440}
# The negative prompt the Wan authors ship with their reference pipeline (it is written in Chinese).
DEFAULT_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
                    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def boundary_step(steps, shift, boundary):
    """First step whose sigma is below the expert boundary, on ComfyUI's 'simple' schedule for a shifted flow model.
    Splitting at steps//2 instead hands the experts the wrong share once shift or step count changes."""
    for i in range(steps):
        t = 1.0 - i / steps
        if shift * t / (1 + (shift - 1) * t) < boundary:
            return max(1, i)
    return steps - 1


class Wan22:
    name = "wan2.2"
    license = "Apache-2.0 (Wan-AI/Wan2.2)"
    notice = ""
    fps = FPS
    native_audio = False
    kind = "video"
    models = (
        [(REPO, f"split_files/diffusion_models/{f}", "diffusion_models") for f in UNET.values()]
        + [(REPO, f"split_files/loras/{f}", "loras") for f in LORAS["v1"].values()]
        + [(repo, prefix + f, "loras") for f, (repo, prefix) in LORA_SOURCES.items()]
        + [(REPO, f"split_files/text_encoders/{TEXT_ENCODER}", "text_encoders"),
           (REPO, f"split_files/vae/{VAE}", "vae"),
           ("Comfy-Org/Frame-Interpolation", f"frame_interpolation/{RIFE}", "frame_interpolation")]
        + finish.MODELS
    )

    @staticmethod
    def frames(seconds):
        # Wan latents are 4x temporally compressed: 4k+1 frames (81 = 5.06 s at 16 fps).
        n = max(17, round(seconds * FPS))
        return n + (1 - n % 4) % 4

    def output_fps(self, spec):
        return FPS * int(self.knobs(spec)["interpolate"])

    @staticmethod
    def knobs(spec):
        return {**R0, **(spec.get("tuning") or {})}

    def model_file(self, spec):
        mode = "i2v" if spec.get("first_frame") else "t2v"
        return UNET[(mode, "high")] + " + " + UNET[(mode, "low")]

    def plan(self, spec, mode):
        """-> list of passes (expert, lora, cfg, start, end) over one shared schedule, plus steps and shift."""
        k = self.knobs(spec)
        q = spec["quality"]
        off = OFFICIAL[mode]
        if q == "full":
            if k["full_sampler"] == "official":
                steps = int(k["full_steps"]) or off["steps"]
                shift = off["shift"]
                split = boundary_step(steps, shift, off["boundary"])
                return steps, shift, "uni_pc", [("high", False, off["cfg_high"], 0, split),
                                                ("low", False, off["cfg_low"], split, 10000)]
            steps, cfg, _ = QUALITY["full"]
            steps = int(k["full_steps"]) or steps
            shift = 8.0
            split = steps // 2 if k["split"] == "half" else boundary_step(steps, shift, off["boundary"])
            return steps, shift, "euler", [("high", False, cfg, 0, split), ("low", False, cfg, split, 10000)]
        shift = float(k["shift_draft"])
        if q == "balanced":
            # Two full-strength base steps at real cfg set the motion, then the LoRA experts finish on an 8-step grid.
            steps = 8
            split = boundary_step(steps, shift, off["boundary"])
            return steps, shift, "euler", [("high", False, 3.5, 0, 2), ("high", True, 1.0, 2, split),
                                           ("low", True, 1.0, split, 10000)]
        steps, cfg, _ = QUALITY[q]
        split = steps // 2 if k["split"] == "half" else boundary_step(steps, shift, off["boundary"])
        return steps, shift, "euler", [("high", True, cfg, 0, split), ("low", True, cfg, split, 10000)]

    def build_graph(self, spec, prep, names, prefix):
        k = self.knobs(spec)
        w, h = SIZES[spec["aspect"]]
        length = self.frames(spec["seconds"])
        first = next((p for kind, p, at in prep["guides"] if kind == "image" and at == 0.0), None)
        last = next((p for kind, p, at in prep["guides"] if kind == "image" and at == "end"), None)
        mode = "i2v" if first else "t2v"
        steps, shift, sampler, passes = self.plan(spec, mode)

        g = {
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": spec["prompt"]}},
            "neg": {"class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["clip", 0], "text": spec.get("negative_prompt") or DEFAULT_NEGATIVE}},
        }
        models = {}
        for expert, lora in dict.fromkeys((e, l) for e, l, *_ in passes):
            key = f"{expert}{'_lora' if lora else ''}"
            g.setdefault(f"unet_{expert}", {"class_type": "UNETLoader",
                                            "inputs": {"unet_name": UNET[(mode, expert)], "weight_dtype": "default"}})
            m = [f"unet_{expert}", 0]
            if lora:
                g[f"lora_{expert}"] = {"class_type": "LoraLoaderModelOnly",
                                       "inputs": {"model": m, "lora_name": LORAS[k["loras"]][(mode, expert)],
                                                  "strength_model": 1.0}}
                m = [f"lora_{expert}", 0]
            g[f"shift_{key}"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": m, "shift": shift}}
            models[(expert, lora)] = [f"shift_{key}", 0]

        if mode == "t2v":
            g["latent"] = {"class_type": "EmptyHunyuanLatentVideo",
                           "inputs": {"width": w, "height": h, "length": length, "batch_size": 1}}
            pos, neg, latent = ["pos", 0], ["neg", 0], ["latent", 0]
        else:
            g["first"] = {"class_type": "LoadImage", "inputs": {"image": names[first]}}
            inputs = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                      "width": w, "height": h, "length": length, "batch_size": 1, "start_image": ["first", 0]}
            if last:
                g["last"] = {"class_type": "LoadImage", "inputs": {"image": names[last]}}
                g["cond"] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {**inputs, "end_image": ["last", 0]}}
            else:
                g["cond"] = {"class_type": "WanImageToVideo", "inputs": inputs}
            pos, neg, latent = ["cond", 0], ["cond", 1], ["cond", 2]

        for i, (expert, lora, cfg, start, end) in enumerate(passes):
            final = i == len(passes) - 1
            g[f"sample{i}"] = {"class_type": "KSamplerAdvanced", "inputs": {
                "noise_seed": spec["seed"], "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": "simple",
                "positive": pos, "negative": neg, "model": models[(expert, lora)],
                "add_noise": "enable" if i == 0 else "disable", "latent_image": latent,
                "start_at_step": start, "end_at_step": end,
                "return_with_leftover_noise": "disable" if final else "enable"}}
            latent = [f"sample{i}", 0]
        g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": latent, "vae": ["vae", 0]}}
        images = finish.upscale(g, ["decode", 0], k, w, h)
        if int(k["interpolate"]) > 1:
            g["rife"] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": RIFE}}
            g["interp"] = {"class_type": "FrameInterpolate",
                           "inputs": {"interp_model": ["rife", 0], "images": images, "multiplier": int(k["interpolate"])}}
            images = ["interp", 0]
        g["video"] = {"class_type": "CreateVideo", "inputs": {"images": images, "fps": self.output_fps(spec)}}
        g["save"] = finish.save_node(spec, prefix)
        return g



BACKEND = Wan22()
