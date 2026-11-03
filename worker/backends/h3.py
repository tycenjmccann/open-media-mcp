"""MiniMax-H3 (video + native audio in one pass), incl. the Singularity v1.3 community fine-tune.

LICENSE: MiniMax H3 Community License. The US, EU, UK and Republic of Korea are Excluded Territories; companies
over USD 20M yearly revenue need written authorization; products must show "Powered by MiniMax H3".
You need that license before you enable it. See THIRD_PARTY_MODELS.md.

Graph = the stock ComfyUI MiniMax-H3 nodes (MiniMaxH3ReferenceToVideo + MiniMaxH3AddGuide), ported 1:1 from the
client we used for production music-video takes.
"""

FPS = 24
SIZES = {"9:16": (768, 1344), "16:9": (1344, 768), "1:1": (1024, 1024), "4:5": (896, 1120)}
MODELS_BY_NAME = {
    "singularity": "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
    "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}
TURBO = {4: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
         8: "minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors"}
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
QUALITY = {"full": 0, "draft8": 8, "draft4": 4}


class H3:
    name = "h3"
    license = "MiniMax H3 Community License (Excluded Territories: US, EU, UK, KR)"
    notice = "Powered by MiniMax H3"
    fps = FPS
    native_audio = True
    kind = "video"
    models = [
        ("WarmBloodAban/Minimax-h3_Singularity", MODELS_BY_NAME["singularity"], "diffusion_models"),
        ("Comfy-Org/MiniMax-H3", "diffusion_models/" + MODELS_BY_NAME["ref2va"], "diffusion_models"),
        ("Comfy-Org/MiniMax-H3", "text_encoders/" + TEXT_ENCODER, "text_encoders"),
        ("Comfy-Org/MiniMax-H3", "vae/" + VIDEO_VAE, "vae"),
        ("Comfy-Org/MiniMax-H3", "vae/" + AUDIO_VAE, "vae"),
        ("Comfy-Org/MiniMax-H3", "loras/" + TURBO[4], "loras"),
        ("lightx2v/Minimax-h3-Turbo", TURBO[8], "loras"),
    ]

    @staticmethod
    def frames(seconds):
        # H3 latent cadence: frame count must be 5 mod 17 (same rule as the Comfy templates).
        n = max(5, round(seconds * FPS))
        return n + (5 - n % 17) % 17

    def model_file(self, spec):
        return MODELS_BY_NAME[spec.get("model") or "singularity"]

    def build_graph(self, spec, prep, names, prefix):
        w, h = SIZES[spec["aspect"]]
        length = self.frames(spec["seconds"])
        turbo = QUALITY[spec["quality"]]
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": self.model_file(spec), "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
            "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        }
        model = ["unet", 0]
        steps = turbo or 20
        if turbo:
            g["lora"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": model, "lora_name": TURBO[turbo], "strength_model": 1.0}}
            model = ["lora", 0]

        r2v = {"clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0], "prompt": spec["prompt"],
               "width": w, "height": h, "length": length, "ref_image_size": "match"}
        for i, path in enumerate(prep["refs"]):
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

        g.update({
            "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": spec["seed"]}},
            "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
            "sched": {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": steps, "denoise": 1.0}},
            "guider": {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": pos}},
            "sample": {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                  "sigmas": ["sched", 0], "latent_image": latent}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
            "decode_audio": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
            "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "audio": ["decode_audio", 0], "fps": FPS}},
            "save": {"class_type": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": f"jobs/{prefix}",
                                                            "format": "mp4", "format.codec": "h264"}},
        })
        return g


BACKEND = H3()
