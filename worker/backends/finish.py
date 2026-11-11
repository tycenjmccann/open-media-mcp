"""Graph pieces shared by the video backends: the optional SeedVR2 upscale pass and the SaveVideo node."""

SEEDVR2 = {"seedvr2_3b": "seedvr2_3b_int8_convrot.safetensors", "seedvr2_7b": "seedvr2_7b_int8_convrot.safetensors"}
SEEDVR2_VAE = "seedvr2_ema_vae_fp16.safetensors"
MODELS = [("Comfy-Org/SeedVR2", f"diffusion_models/{f}", "diffusion_models") for f in SEEDVR2.values()] + \
         [("Comfy-Org/SeedVR2", f"vae/{SEEDVR2_VAE}", "vae")]


def upscaled_size(w, h, short_edge):
    s = short_edge / min(w, h)
    return int(round(w * s / 16)) * 16, int(round(h * s / 16)) * 16


def upscale(g, images, knobs, w, h):
    """SeedVR2 one-step restoration to `upscale_height` on the short edge (the Comfy-Org template graph, untrimmed).
    Runs before frame interpolation so it sees the model's native frames, not interpolated ones."""
    name = knobs.get("upscale", "none")
    if name in (None, "none"):
        return images
    tw, th = upscaled_size(w, h, int(knobs.get("upscale_height") or 1440))
    g.update({
        "sr_unet": {"class_type": "UNETLoader", "inputs": {"unet_name": SEEDVR2[name], "weight_dtype": "default"}},
        "sr_vae": {"class_type": "VAELoader", "inputs": {"vae_name": SEEDVR2_VAE}},
        "sr_resize": {"class_type": "ImageScale", "inputs": {"image": images, "upscale_method": "lanczos",
                                                             "width": tw, "height": th, "crop": "disabled"}},
        "sr_pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["sr_resize", 0]}},
        "sr_enc": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["sr_pre", 0], "vae": ["sr_vae", 0],
                                                              "tile_size": 512, "overlap": 128, "temporal_size": 64,
                                                              "temporal_overlap": 8}},
        "sr_cond": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["sr_unet", 0], "vae_conditioning": ["sr_enc", 0]}},
        "sr_sample": {"class_type": "KSampler", "inputs": {"model": ["sr_unet", 0], "positive": ["sr_cond", 0],
                                                           "negative": ["sr_cond", 1], "latent_image": ["sr_enc", 0],
                                                           "seed": 1, "steps": 1, "cfg": 1.0, "sampler_name": "euler",
                                                           "scheduler": "simple", "denoise": 1.0}},
        "sr_dec": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["sr_sample", 0], "vae": ["sr_vae", 0],
                                                              "tile_size": 512, "overlap": 128, "temporal_size": 64,
                                                              "temporal_overlap": 8}},
        "sr_post": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["sr_dec", 0],
                                                                      "original_resized_images": ["sr_resize", 0],
                                                                      "color_correction_method": "lab"}},
    })
    return ["sr_post", 0]


def save_node(spec, prefix, video=("video", 0)):
    # "hq": ComfyUI writes near-lossless ProRes and the worker does the one lossy x264 encode itself.
    if (spec.get("tuning") or {}).get("encode") == "hq":
        fmt = {"format": "mov", "format.codec": "prores"}
    else:
        fmt = {"format": "mp4", "format.codec": "h264"}
    return {"class_type": "SaveVideo", "inputs": {"video": list(video), "filename_prefix": f"jobs/{prefix}", **fmt}}
