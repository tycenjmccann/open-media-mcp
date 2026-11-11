"""Server-side prompt rewrite (tuning knob enhance_prompt), run on the worker before the render.

Both model makers say a rewrite step matters: Wan ships a prompt extender in its reference pipeline, and MiniMax's
hosted H3 runs every prompt through a rewriter (H3-Context-IR) that isn't part of the open weights. We reproduce
each with an LLM on Amazon Bedrock:
  * wan2.2: Wan's own extender system prompts (worker/prompts/wan_*.txt, Apache-2.0, vendored).
  * h3: MiniMax's published h3-prompt-writing guides, fetched at boot from their repo at a pinned commit
    (MEDIA_PROMPTS_DIR) rather than vendored, because they ship under the MiniMax H3 Community License.
  * h3-image: the instructions of MiniMax-H3-Image-PE (MiniMax's local stand-in for its hosted H3-Image-Context-IR),
    fetched the same way. The model card says the DiT expects a detailed English caption of the whole image; the
    expander also picks the canvas ratio, which the worker uses when the caller left aspect on auto.
The original and rewritten prompts both land in the provenance manifest.
"""
import io
import json
import os

import boto3
from botocore.config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MEDIA_PROMPT_MODEL", "us.anthropic.claude-sonnet-5-5")
GUIDES = os.environ.get("MEDIA_PROMPTS_DIR", "/opt/media/prompts")
_brt = None


def _client():
    global _brt
    if _brt is None:
        _brt = boto3.client("bedrock-runtime", config=Config(read_timeout=300, retries={"max_attempts": 6, "mode": "adaptive"}))
    return _brt


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _image_block(path, max_side=1024):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return {"image": {"format": "jpeg", "source": {"bytes": buf.getvalue()}}}


def _labels(spec, prep, keyframe_refs):
    """The reference labels the H3 text encoder will emit, in its order, so the rewrite can use the right tags."""
    pics = [("reference image", p) for p in prep["refs"]]
    if keyframe_refs:
        for kind, p, at in prep["guides"]:
            if kind == "image":
                where = "first frame" if at == 0.0 else "last frame" if at == "end" else f"keyframe at {at} s"
                pics.append((where, p))
    lines, images = [], []
    for i, (role, p) in enumerate(pics[:9], 1):
        lines.append(f"<Picture {i}>: {role} (attached below as Picture {i})")
        images.append((f"Picture {i}", p))
    a = 0
    for i, (_, has_audio) in enumerate(prep["ref_videos"], 1):
        if has_audio:
            a += 1
            lines.append(f"<Audio {a}>: the soundtrack of <Video {i}>")
        lines.append(f"<Video {i}>: reference video")
    for _ in prep["ref_audios"]:
        a += 1
        lines.append(f"<Audio {a}>: reference audio (voice or sound to take timbre from)")
    if prep["soundtrack"]:
        lines.append("A soundtrack is supplied as an audio guide on the timeline: on-camera singing or speech must lip-sync "
                     "to it, so write the sung/spoken words in <d> tags exactly as the user gave them, and set "
                     "non_diegetic_music to N/A unless the user asked for extra score.")
    return lines, images


def _h3_system(spec, prep, keyframe_refs):
    has_refs = bool(prep["refs"] or prep["ref_videos"] or prep["ref_audios"]) or \
        (keyframe_refs and any(k == "image" for k, _, _ in prep["guides"]))
    guide = "ref-en.txt" if has_refs else "base-en.txt"
    path = os.path.join(GUIDES, guide)
    body = _read(path) if os.path.exists(path) else ""
    mode = "Ref2VA (full-reference): use the six-section format" if has_refs else \
        "a base mode (T2VA/I2VA/FL2VA/L2VA as the inputs imply): use the instruction + three-field format"
    return (f"You rewrite a user's video request into a MiniMax H3 prompt. This request is {mode} from the guide below. "
            "Follow the guide's field names, section order, label rules, timing notation and length guidance exactly. "
            "Keep the user's subjects, actions, dialogue and lyrics; do not invent new characters or change quoted words. "
            "Write in English; keep dialogue, lyrics and on-screen text in their original language. "
            "Reply with the final prompt only: no preamble, no code fences, no commentary.\n\n"
            f"<guide>\n{body}\n</guide>")


def _wan_system(spec, prep):
    i2v = any(k == "image" and at == 0.0 for k, _, at in prep["guides"])
    return _read(os.path.join(HERE, "prompts", "wan_i2v.txt" if i2v else "wan_t2v.txt")) + \
        "\n\nReply with the rewritten prompt only, in English, with no preamble."


def _json_reply(text):
    """The expander answers one JSON object; tolerate prose or a fence around it."""
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the rewrite")
        return json.loads(text[start:end + 1])


def _h3image_request(spec, prep):
    """-> (system, content) in MiniMax-H3-Image-PE's own request shape (prompt/t2i and prompt/edit of that repo)."""
    from backends.h3image import image_ratio
    aspect = spec.get("aspect") or "auto"
    if not prep["refs"]:
        text = spec["prompt"] if aspect == "auto" else f"{spec['prompt']}\n\nFinal canvas aspect ratio: {aspect}."
        system = _read(os.path.join(GUIDES, "h3image_t2i_system_prompt.txt"))
        task = _read(os.path.join(GUIDES, "h3image_t2i_task_prompt.txt"))
        return system, [{"text": task.format(text)}]
    content = []
    for i, path in enumerate(prep["refs"][:16], 1):
        w, h = image_ratio(path)
        content += [{"text": f"Reference Media: <Picture {i}>, aspect_ratio {w}:{h}"}, _image_block(path)]
    content.append({"text": f"User Editing Instruction (JSON string): {json.dumps(spec['prompt'], ensure_ascii=False)}\n"
                            f"User Aspect Ratio Selection: {'adaptive' if aspect == 'auto' else aspect}"})
    return _read(os.path.join(GUIDES, "h3image_edit_task_prompt.txt")), content


def _enhance_h3image(spec, prep):
    system, content = _h3image_request(spec, prep)
    r = _client().converse(
        modelId=MODEL,
        system=[{"text": system + "\n\nReply with the JSON object only."}, {"cachePoint": {"type": "default"}}],
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 8000},
    )
    out = _json_reply("".join(c.get("text", "") for c in r["output"]["message"]["content"]).strip())
    caption = str(out.get("expanded_caption") or "").strip()
    if not caption:
        raise ValueError("empty expanded_caption")
    usage = r.get("usage", {})
    return caption[:16000], {"model": MODEL, "input_tokens": usage.get("inputTokens"),
                             "output_tokens": usage.get("outputTokens"),
                             "cache_read_tokens": usage.get("cacheReadInputTokens"), "original": spec["prompt"],
                             "aspect": out.get("aspect_ratio") or out.get("target_aspect_ratio")}


def enhance(spec, prep, backend_name, keyframe_refs=False):
    """-> (prompt, info). Never fails the job: on any error the original prompt is used and the error recorded."""
    try:
        if backend_name == "h3-image":
            return _enhance_h3image(spec, prep)
        if backend_name == "h3":
            system = _h3_system(spec, prep, keyframe_refs)
            lines, images = _labels(spec, prep, keyframe_refs)
        else:
            system = _wan_system(spec, prep)
            first = next((p for k, p, at in prep["guides"] if k == "image" and at == 0.0), None)
            lines, images = [], ([("First frame", first)] if first else [])
        content = [{"text": f"Target video: {spec['seconds']} seconds, aspect {spec['aspect']}.\n"
                            + ("Inputs and their labels:\n" + "\n".join(lines) + "\n" if lines else "")
                            + f"\nUser request:\n{spec['prompt']}"}]
        for label, path in images:
            content += [{"text": label + ":"}, _image_block(path)]
        r = _client().converse(
            modelId=MODEL,
            system=[{"text": system}, {"cachePoint": {"type": "default"}}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 4000},
        )
        text = "".join(c.get("text", "") for c in r["output"]["message"]["content"]).strip()
        if not text:
            raise ValueError("empty rewrite")
        usage = r.get("usage", {})
        return text[:12000], {"model": MODEL, "input_tokens": usage.get("inputTokens"),
                              "output_tokens": usage.get("outputTokens"),
                              "cache_read_tokens": usage.get("cacheReadInputTokens"), "original": spec["prompt"]}
    except Exception as exc:  # the render still runs on the user's own prompt
        return spec["prompt"], {"model": MODEL, "error": f"{type(exc).__name__}: {exc}"[:500], "original": spec["prompt"]}
