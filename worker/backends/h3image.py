"""MiniMax-H3-Image (MiniMax): 33B single-stream DiT from the H3 family, text-to-image and editing in one checkpoint,
about 16 MP at any aspect, up to 16 reference images (<Picture 1> ... <Picture 16>), distilled to 6 steps, no CFG.

LICENSE: MiniMax-H3-Image Community License. The US, EU, UK and Republic of Korea are Excluded Territories; written
authorization is needed above USD 1M annualized revenue or to offer it as a hosted service to third parties;
commercial products must show "MiniMax-H3-Image". You need that license before you enable it. See THIRD_PARTY_MODELS.md.

Graph = the stock ComfyUI template (Comfy-Org/workflow_templates image_minimax_h3_image_t2i / _edit): int8 DiT, the
same Qwen3-VL text encoder file as the h3 video backend, MiniMaxH3ImageConditioning, KSampler 6 steps / cfg 1 /
euler / simple. The Comfy node scales the references itself (short edge 1024-2048), so they go in as uploaded.
count > 1 adds one sampler chain per image on the shared model and conditioning, image i seeded seed + i.
"""
import math
import re

UNET = "minimax_h3_image_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE = "minimax_h3_image_vae_fp16.safetensors"
STEPS, CFG = 6, 1.0
MAX_REFS = 16
# quality -> target area. MiniMax: works from about 4 MP to 20 MP, best at about 16 MP.
AREA = {"full": 4096 * 4096, "draft": 2048 * 2048}
RATIO = re.compile(r"^\s*(\d{1,3})\s*:\s*(\d{1,3})\s*$")


def size_for(ratio, area):
    """MiniMax's own rule (examples/context_ir.py): about `area` pixels at the ratio, both sides multiples of 32."""
    x, y = ratio
    return round(math.sqrt(area * x / y) / 32) * 32, round(math.sqrt(area * y / x) / 32) * 32


def parse_ratio(text):
    m = RATIO.match(str(text or ""))
    if not m or not int(m[1]) or not int(m[2]) or not 1 / 4 <= int(m[1]) / int(m[2]) <= 4:
        return None
    return int(m[1]), int(m[2])


def image_ratio(path):
    from PIL import Image, ImageOps  # a ComfyUI requirement, so in the worker venv
    with Image.open(path) as im:
        w, h = ImageOps.exif_transpose(im).size
    g = math.gcd(w, h)
    return w // g, h // g


class H3Image:
    name = "h3-image"
    kind = "image"
    license = "MiniMax-H3-Image Community License (Excluded Territories: US, EU, UK, KR)"
    notice = "Powered by MiniMax-H3-Image"
    fps = None
    native_audio = False
    models = [("Comfy-Org/MiniMax-H3-Image", f"diffusion_models/{UNET}", "diffusion_models"),
              ("Comfy-Org/MiniMax-H3", f"text_encoders/{TEXT_ENCODER}", "text_encoders"),
              ("Comfy-Org/MiniMax-H3-Image", f"vae/{VAE}", "vae")]

    @staticmethod
    def frames(seconds):
        return None

    def model_file(self, spec):
        return UNET

    @staticmethod
    def canvas(spec, prep):
        """aspect 'auto' (no rewrite picked one) = the edited image's own ratio, or square for text-to-image."""
        ratio = parse_ratio(spec.get("aspect"))
        if ratio is None:
            ratio = image_ratio(prep["refs"][0]) if prep["refs"] else (1, 1)
        return size_for(ratio, AREA[spec.get("quality") or "full"])

    def build_graph(self, spec, prep, names, prefix):
        w, h = self.canvas(spec, prep)
        cond = {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": spec["prompt"], "width": w, "height": h}
        g = {
            "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        }
        for i, path in enumerate(prep["refs"][:MAX_REFS], 1):
            g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": names[path]}}
            cond[f"images.image_{i}"] = [f"ref{i}", 0]
        g["cond"] = {"class_type": "MiniMaxH3ImageConditioning", "inputs": cond}
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["cond", 0]}}
        for i in range(int(spec.get("count") or 1)):
            n = "" if i == 0 else f"_{i + 1}"
            g["ks" + n] = {"class_type": "KSampler", "inputs": {
                "model": ["unet", 0], "positive": ["cond", 0], "negative": ["neg", 0], "latent_image": ["cond", 1],
                "seed": spec["seed"] + i, "steps": STEPS, "cfg": CFG, "sampler_name": "euler", "scheduler": "simple",
                "denoise": 1.0}}
            g["dec" + n] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks" + n, 0], "vae": ["vae", 0]}}
            g["save" + n] = {"class_type": "SaveImage", "inputs": {"images": ["dec" + n, 0], "filename_prefix": f"jobs/{prefix}_{i + 1}"}}
        return g


BACKEND = H3Image()
