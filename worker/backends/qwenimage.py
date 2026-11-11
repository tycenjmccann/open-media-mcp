"""Qwen-Image (Qwen team, Alibaba): 20B MMDiT text to image, strong at photoreal scenes and at lettering inside
the picture (English and Chinese).

LICENSE: Apache-2.0 (Qwen/Qwen-Image; the Qwen2.5-VL-7B text encoder is Apache-2.0 too). No attribution notice.

Graph = the ComfyUI Qwen-Image text-to-image graph on the fp8 repack (Comfy-Org/Qwen-Image_ComfyUI): AuraFlow
sampling shift 3.1, euler/simple, cfg 3.0, 30 steps. count > 1 adds one sampler chain per image on the shared model
and prompt encodings, image i seeded seed + i, so any one of them re-renders on its own with count 1.
Sizes are Qwen's own ~1.6 MP aspect table; 1140 is not a multiple of 8, so 4:3 / 3:4 come out 1472x1136 / 1136x1472
(the worker records the real size).
"""

REPO = "Comfy-Org/Qwen-Image_ComfyUI"
UNET = "qwen_image_fp8_e4m3fn.safetensors"
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
SIZES = {"16:9": (1664, 928), "9:16": (928, 1664), "1:1": (1328, 1328), "4:3": (1472, 1140), "3:4": (1140, 1472)}
NEGATIVE = "blurry, lowres, deformed, watermark"
STEPS, CFG, SHIFT = 30, 3.0, 3.1


class QwenImage:
    name = "qwen-image"
    kind = "image"
    license = "Apache-2.0 (Qwen/Qwen-Image)"
    notice = ""
    fps = None
    native_audio = False
    models = [(REPO, f"split_files/diffusion_models/{UNET}", "diffusion_models"),
              (REPO, f"split_files/text_encoders/{TEXT_ENCODER}", "text_encoders"),
              (REPO, f"split_files/vae/{VAE}", "vae")]

    @staticmethod
    def frames(seconds):
        return None

    def model_file(self, spec):
        return UNET

    def build_graph(self, spec, prep, names, prefix):
        w, h = SIZES[spec.get("aspect") or "16:9"]
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
            "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": SHIFT}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": spec["prompt"]}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": spec.get("negative_prompt") or NEGATIVE}},
            "lat": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        }
        for i in range(int(spec.get("count") or 1)):
            n = "" if i == 0 else f"_{i + 1}"
            g["ks" + n] = {"class_type": "KSampler", "inputs": {
                "model": ["shift", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["lat", 0],
                "seed": spec["seed"] + i, "steps": int(spec.get("steps") or STEPS), "cfg": float(spec.get("cfg") or CFG),
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
            g["dec" + n] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks" + n, 0], "vae": ["vae", 0]}}
            g["save" + n] = {"class_type": "SaveImage", "inputs": {"images": ["dec" + n, 0], "filename_prefix": f"jobs/{prefix}_{i + 1}"}}
        return g


BACKEND = QwenImage()
