"""MiniMax-H3 (video + native audio in one pass), incl. the Singularity v1.3 community fine-tune.

LICENSE: MiniMax H3 Community License. The US, EU, UK and Republic of Korea are Excluded Territories; companies
over USD 20M yearly revenue need written authorization; products must show "Powered by MiniMax H3".
You need that license before you enable it. See THIRD_PARTY_MODELS.md.

Graph = the stock ComfyUI MiniMax-H3 nodes (MiniMaxH3ReferenceToVideo + MiniMaxH3AddGuide). Knobs come from
spec["tuning"] (a named recipe resolved by the tools Lambda); missing knobs fall back to recipe r0, the graph we
first shipped, so old jobs render unchanged.
"""
from . import finish

FPS = 24
SIZES = {"r0": {"9:16": (768, 1344), "16:9": (1344, 768), "1:1": (1024, 1024), "4:5": (896, 1120)},
         # MiniMax's own canvases: 768 on the short edge, area capped at 1344x768.
         "official": {"9:16": (768, 1344), "16:9": (1344, 768), "1:1": (768, 768), "4:5": (768, 960)}}
MODELS_BY_NAME = {
    "singularity": "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
    "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}
# quality -> (LoRA file, video shift, audio shift). The 768p turbo LoRAs were distilled at video shift 6.
TURBO = {
    "r0": {4: ("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", None, None),
           8: ("minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", None, None)},
    "768p": {4: ("minimax_h3_ref2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", 6.0, 3.0),
             8: ("minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors", 6.0, 3.0)},
}
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
QUALITY = {"full": 0, "draft8": 8, "draft4": 4}
# Open-H3-Regenerate-2K's refine schedule: re-noise the upscaled latent to ~0.91 and walk back down in 3 steps.
REGEN_SIGMAS = "0.909375, 0.857143, 0.666667, 0.0"
R0 = {"turbo": "r0", "canvas": "r0", "keyframe_refs": False, "steps": 20, "sampler": "res_multistep",
      "ref_size": "match", "regen_2k": False, "upscale": "none", "upscale_height": 1440}
MAX_REF_IMAGES = 9


class H3:
    name = "h3"
    license = "MiniMax H3 Community License (Excluded Territories: US, EU, UK, KR)"
    notice = "Powered by MiniMax H3"
    fps = FPS
    native_audio = True
    kind = "video"
    # The audio VAE crops guide audio that isn't a whole number of 25 ms frames (800 samples at 32 kHz).
    audio_frame_s = 0.025
    models = [
        ("WarmBloodAban/Minimax-h3_Singularity", MODELS_BY_NAME["singularity"], "diffusion_models"),
        ("Comfy-Org/MiniMax-H3", "diffusion_models/" + MODELS_BY_NAME["ref2va"], "diffusion_models"),
        ("Comfy-Org/MiniMax-H3", "text_encoders/" + TEXT_ENCODER, "text_encoders"),
        ("Comfy-Org/MiniMax-H3", "vae/" + VIDEO_VAE, "vae"),
        ("Comfy-Org/MiniMax-H3", "vae/" + AUDIO_VAE, "vae"),
        ("Comfy-Org/MiniMax-H3", "loras/" + TURBO["r0"][4][0], "loras"),
        ("lightx2v/Minimax-h3-Turbo", TURBO["r0"][8][0], "loras"),
        ("lightx2v/Minimax-h3-Turbo", TURBO["768p"][4][0], "loras"),
        ("LBH-123-AI/Minimax_h3_latent_Upscaler", "minimax_h3_latent_upscaler_3d_conv_v1/" + LATENT_UPSCALER,
         "latent_upscale_models"),
    ] + finish.MODELS

    @staticmethod
    def frames(seconds):
        # H3 latent cadence: frame count must be 5 mod 17 (same rule as the Comfy templates).
        n = max(5, round(seconds * FPS))
        return n + (5 - n % 17) % 17

    def output_fps(self, spec):
        return FPS

    @staticmethod
    def knobs(spec):
        return {**R0, **(spec.get("tuning") or {})}

    def model_file(self, spec):
        return MODELS_BY_NAME[spec.get("model") or "singularity"]

    def build_graph(self, spec, prep, names, prefix):
        k = self.knobs(spec)
        w, h = SIZES[k["canvas"]][spec["aspect"]]
        length = self.frames(spec["seconds"])
        turbo = QUALITY[spec["quality"]]
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": self.model_file(spec), "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
            "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        }
        base = ["unet", 0]
        model = base
        steps = turbo or int(k["steps"])
        if turbo:
            lora, sv, sa = TURBO[k["turbo"]][turbo]
            g["lora"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": model, "lora_name": lora, "strength_model": 1.0}}
            model = ["lora", 0]
            if sv:
                g["shift"] = {"class_type": "MiniMaxH3SigmaShift",
                              "inputs": {"model": model, "shift_video": sv, "shift_audio": sa}}
                model = ["shift", 0]

        # The turbo LoRAs were trained with "match"; "max" (no downscale of large refs) is for full renders.
        ref_size = k["ref_size"] if not turbo else "match"
        r2v = {"clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0], "prompt": spec["prompt"],
               "width": w, "height": h, "length": length, "ref_image_size": ref_size}
        ref_images = list(prep["refs"])
        if k["keyframe_refs"]:
            # The text encoder only sees images wired as references, so <Picture N> tags for first_frame / keyframes /
            # last_frame resolve to nothing unless those stills are references too (numbering continues after refs).
            ref_images += [p for kind, p, _ in prep["guides"] if kind == "image"]
        for i, path in enumerate(ref_images[:MAX_REF_IMAGES]):
            g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": names[path]}}
            r2v[f"ref_images.ref_image_{i}"] = [f"ref{i}", 0]
        paired = True  # soundtracks pair by index, so stop pairing at the first silent video
        for i, (path, has_audio) in enumerate(prep["ref_videos"]):
            g[f"rv{i}"] = {"class_type": "LoadVideo", "inputs": {"file": names[path]}}
            g[f"rvc{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"rv{i}", 0]}}
            r2v[f"ref_videos.ref_video_{i}"] = [f"rvc{i}", 0]
            paired = paired and has_audio
            if paired:
                r2v[f"ref_video_audios.ref_video_audio_{i}"] = [f"rvc{i}", 1]
        for i, path in enumerate(prep["ref_audios"]):
            g[f"ra{i}"] = {"class_type": "LoadAudio", "inputs": {"audio": names[path]}}
            r2v[f"ref_audios.ref_audio_{i}"] = [f"ra{i}", 0]
        g["r2v"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": r2v}
        pos, latent = ["r2v", 0], ["r2v", 1]

        def guide(key, extra, at):
            nonlocal pos
            idx = -1 if at == "end" else (-max(1, round(-at * FPS)) if at < 0 else min(round(at * FPS), length - 1))
            g[key] = {"class_type": "MiniMaxH3AddGuide",
                      "inputs": {"positive": pos, "latent": latent, "vae": ["vae", 0], "audio_vae": ["avae", 0],
                                 "frame_idx": idx, **extra}}
            pos = [key, 0]  # guides only rewrite conditioning; the latent passes through untouched

        for j, (kind, path, at) in enumerate(prep["guides"]):
            if kind == "image":
                g[f"kf{j}"] = {"class_type": "LoadImage", "inputs": {"image": names[path]}}
                guide(f"kfg{j}", {"image": [f"kf{j}", 0]}, at)
            else:
                g[f"kf{j}"] = {"class_type": "LoadVideo", "inputs": {"file": names[path]}}
                g[f"kfc{j}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"kf{j}", 0]}}
                guide(f"kfg{j}", {"image": [f"kfc{j}", 0]}, at)
        if prep["soundtrack"]:
            g["track"] = {"class_type": "LoadAudio", "inputs": {"audio": names[prep["soundtrack"][0]]}}
            guide("track_guide", {"audio": ["track", 0]}, (spec.get("soundtrack") or {}).get("at", 0.0))

        sampler = k["sampler"] if not turbo else "res_multistep"
        g.update({
            "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": spec["seed"]}},
            "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
            "sched": {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}},
            "guider": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": pos}},
            "sample": {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                  "sigmas": ["sched", 0], "latent_image": latent}},
            "decode_audio": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        })
        video_latent = ["sample", 0]
        if k["regen_2k"]:
            # Latent upscale + a short re-denoise at the target size (the open take on MiniMax's hosted 2K pass).
            # Audio is decoded from the first pass; the refine pass only exists to add picture detail.
            tw, th = finish.upscaled_size(w, h, int(k["upscale_height"] or 1440))
            g.update({
                "lup_model": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LATENT_UPSCALER}},
                "lup": {"class_type": "MiniMaxH3LatentUpscaleWithModel",
                        "inputs": {"model": ["lup_model", 0], "samples": ["sample", 0], "width": tw // 32 * 32,
                                   "height": th // 32 * 32}},
                "regen_sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": REGEN_SIGMAS}},
                "regen_guider": {"class_type": "BasicGuider", "inputs": {"model": base, "conditioning": ["r2v", 0]}},
                "regen_sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
                "regen": {"class_type": "SamplerCustomAdvanced",
                          "inputs": {"noise": ["noise", 0], "guider": ["regen_guider", 0], "sampler": ["regen_sampler", 0],
                                     "sigmas": ["regen_sigmas", 0], "latent_image": ["lup", 0]}},
            })
            video_latent = ["regen", 0]
        g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": video_latent, "vae": ["vae", 0]}}
        images = finish.upscale(g, ["decode", 0], k, w, h) if not k["regen_2k"] else ["decode", 0]
        g["video"] = {"class_type": "CreateVideo", "inputs": {"images": images, "audio": ["decode_audio", 0], "fps": FPS}}
        g["save"] = finish.save_node(spec, prefix)
        return g


BACKEND = H3()
