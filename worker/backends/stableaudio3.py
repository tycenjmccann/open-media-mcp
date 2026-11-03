"""Stable Audio 3 (Stability AI): text to stereo music, instruments and sound effects, from one-shots to minutes.

LICENSE: Stability AI Community License. Free, including commercial use (register at stability.ai/community-license),
for individuals and organizations under USD 1M annual revenue; above that you need a Stability AI enterprise license.
Products built on it must display "Powered by Stability AI". See THIRD_PARTY_MODELS.md.

Graph = the stock ComfyUI Stable Audio 3 template (Comfy-Org/workflow_templates audio_stable_audio_3_medium), minus its
optional Qwen prompt-expansion branch: the post-trained checkpoints sample in 8 LCM steps at cfg 1.
"""

REPO = "Comfy-Org/stable-audio-3"
CHECKPOINTS = {
    "medium": "stable_audio_3_medium.safetensors",            # music + SFX, best quality
    "small-music": "stable_audio_3_small_music.safetensors",  # faster, music
    "small-sfx": "stable_audio_3_small_sfx.safetensors",      # faster, sound effects
}
TEXT_ENCODER = "t5gemma_b_b_ul2.safetensors"


class StableAudio3:
    name = "stable-audio-3"
    kind = "audio"
    license = "Stability AI Community License (free under USD 1M annual revenue; enterprise license above)"
    notice = "Powered by Stability AI"
    fps = None
    native_audio = True
    models = [(REPO, f"checkpoints/{f}", "checkpoints") for f in CHECKPOINTS.values()] + \
             [(REPO, f"text_encoders/{TEXT_ENCODER}", "text_encoders")]

    @staticmethod
    def frames(seconds):
        return None

    def model_file(self, spec):
        return CHECKPOINTS[spec.get("model") or "medium"]

    def build_graph(self, spec, prep, names, prefix):
        return {
            "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.model_file(spec)}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "stable_audio", "device": "default"}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": spec["prompt"]}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": spec.get("negative_prompt") or ""}},
            "latent": {"class_type": "EmptyLatentAudio", "inputs": {"seconds": float(spec["seconds"]), "batch_size": 1}},
            "sample": {"class_type": "KSampler", "inputs": {
                "model": ["ckpt", 0], "seed": spec["seed"], "steps": 8, "cfg": 1.0, "sampler_name": "lcm",
                "scheduler": "simple", "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}},
            "decode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
            "save": {"class_type": "SaveAudio", "inputs": {"audio": ["decode", 0], "filename_prefix": f"jobs/{prefix}"}},
        }


BACKEND = StableAudio3()
