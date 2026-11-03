"""Wan 2.2 A14B (Wan-AI, Apache-2.0): text-to-video and image-to-video, 720p, 16 fps, silent output.

Two-expert MoE: a high-noise model runs the first half of the steps, a low-noise model the second half
(two KSamplerAdvanced passes), as in the stock ComfyUI Wan 2.2 14B templates. Draft quality adds the lightx2v
4-step distillation LoRAs (cfg 1). A soundtrack is muxed onto the silent render (no lip-sync on this backend).
"""

FPS = 16
SIZES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960), "4:5": (768, 960)}
REPO = "Comfy-Org/Wan-2.2"
UNET = {
    ("t2v", "high"): "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    ("t2v", "low"): "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    ("i2v", "high"): "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    ("i2v", "low"): "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
}
LORA = {
    ("t2v", "high"): "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
    ("t2v", "low"): "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
    ("i2v", "high"): "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    ("i2v", "low"): "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
}
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
# quality -> (steps, cfg, use the 4-step LoRAs)
QUALITY = {"full": (20, 3.5, False), "draft8": (8, 1.0, True), "draft4": (4, 1.0, True)}
# The negative prompt the Wan authors ship with their reference pipeline (it is written in Chinese).
DEFAULT_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
                    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


class Wan22:
    name = "wan2.2"
    license = "Apache-2.0 (Wan-AI/Wan2.2)"
    notice = ""
    fps = FPS
    native_audio = False
    kind = "video"
    models = (
        [(REPO, f"split_files/diffusion_models/{f}", "diffusion_models") for f in UNET.values()]
        + [(REPO, f"split_files/loras/{f}", "loras") for f in LORA.values()]
        + [(REPO, f"split_files/text_encoders/{TEXT_ENCODER}", "text_encoders"),
           (REPO, f"split_files/vae/{VAE}", "vae")]
    )

    @staticmethod
    def frames(seconds):
        # Wan latents are 4x temporally compressed: 4k+1 frames (81 = 5.06 s at 16 fps).
        n = max(17, round(seconds * FPS))
        return n + (1 - n % 4) % 4

    def model_file(self, spec):
        mode = "i2v" if spec.get("first_frame") else "t2v"
        return UNET[(mode, "high")] + " + " + UNET[(mode, "low")]

    def build_graph(self, spec, prep, names, prefix):
        w, h = SIZES[spec["aspect"]]
        length = self.frames(spec["seconds"])
        steps, cfg, lora = QUALITY[spec["quality"]]
        first = next((p for kind, p, at in prep["guides"] if kind == "image" and at == 0.0), None)
        last = next((p for kind, p, at in prep["guides"] if kind == "image" and at == "end"), None)
        mode = "i2v" if first else "t2v"

        g = {
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": spec["prompt"]}},
            "neg": {"class_type": "CLIPTextEncode",
                    "inputs": {"clip": ["clip", 0], "text": spec.get("negative_prompt") or DEFAULT_NEGATIVE}},
        }
        models = {}
        for expert in ("high", "low"):
            g[f"unet_{expert}"] = {"class_type": "UNETLoader",
                                   "inputs": {"unet_name": UNET[(mode, expert)], "weight_dtype": "default"}}
            m = [f"unet_{expert}", 0]
            if lora:
                g[f"lora_{expert}"] = {"class_type": "LoraLoaderModelOnly",
                                       "inputs": {"model": m, "lora_name": LORA[(mode, expert)], "strength_model": 1.0}}
                m = [f"lora_{expert}", 0]
            g[f"shift_{expert}"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": m, "shift": 8.0}}
            models[expert] = [f"shift_{expert}", 0]

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

        split = steps // 2
        common = {"noise_seed": spec["seed"], "steps": steps, "cfg": cfg, "sampler_name": "euler",
                  "scheduler": "simple", "positive": pos, "negative": neg}
        g["sample_high"] = {"class_type": "KSamplerAdvanced", "inputs": {
            **common, "model": models["high"], "add_noise": "enable", "latent_image": latent,
            "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}}
        g["sample_low"] = {"class_type": "KSamplerAdvanced", "inputs": {
            **common, "model": models["low"], "add_noise": "disable", "latent_image": ["sample_high", 0],
            "start_at_step": split, "end_at_step": 10000, "return_with_leftover_noise": "disable"}}
        g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample_low", 0], "vae": ["vae", 0]}}
        g["video"] = {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": FPS}}
        g["save"] = {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": f"jobs/{prefix}",
                                                          "format": "mp4", "format.codec": "h264"}}
        return g


BACKEND = Wan22()
