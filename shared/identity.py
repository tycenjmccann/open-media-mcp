"""Character identity check on Amazon Rekognition, shared by the tools Lambda (check_identity) and the GPU worker
(generate_video identity_check).

Every face in every reference image (each panel of a character sheet, plus any approved images) is compared with
the largest faces in the image being checked; the image's score is its best match (0-100). Verdicts:

  PASS    score >= pass_at (97) on a clear face
  FAIL    score <  review_at (90) on a clear face
  REVIEW  in between, or the face is under min_face px tall, or covered (sunglasses, visor, hands): Rekognition's
          score is unreliable there in both directions, so a person decides

Calibrated 2026-11-05 on 17 AI images: 9 of one character, 3 of a look-alike character and 5 different women given
the character's hair, makeup and outfits. No wrong PASS or FAIL; every look-alike scored <= 92.4, clear-face matches
of the character 95.8-99.8, and faces under ~100 px scored anywhere from 21 to 94 for the real character.

Faces are cropped with a margin and enlarged before comparing, which needs Pillow.
"""
import io
from concurrent.futures import ThreadPoolExecutor

PASS_AT, REVIEW_AT, MIN_FACE = 97.0, 90.0, 100
MAX_FACES = 3  # largest faces checked per image; background people beyond that are ignored
MAX_REFERENCE_FACES = 12


def _jpeg(im, quality=95):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def faces(rek, data, margin=0.6, face_px=400, min_conf=90.0):
    """Every confident face, largest first: an enlarged JPEG crop plus what decides whether its score is trusted."""
    from PIL import Image

    src = Image.open(io.BytesIO(data))
    fmt, im = src.format, src.convert("RGB")
    if fmt not in ("JPEG", "PNG") or len(data) > 5 * 1024 * 1024:  # Rekognition takes JPEG/PNG up to 5 MB inline
        data = _jpeg(im, 92)
    found = rek.detect_faces(Image={"Bytes": data}, Attributes=["ALL"])["FaceDetails"]
    W, H = im.size
    out = []
    for f in sorted(found, key=lambda f: -f["BoundingBox"]["Width"] * f["BoundingBox"]["Height"]):
        if f["Confidence"] < min_conf:
            continue
        b = f["BoundingBox"]
        x, y, w, h = b["Left"] * W, b["Top"] * H, b["Width"] * W, b["Height"] * H
        c = im.crop((max(0, int(x - w * margin)), max(0, int(y - h * margin)),
                     min(W, int(x + w * (1 + margin))), min(H, int(y + h * (1 + margin)))))
        if h < face_px:
            k = face_px / h
            c = c.resize((int(c.width * k), int(c.height * k)), Image.LANCZOS)
        out.append({"crop": _jpeg(c), "face_px": round(h),
                    "occluded": bool(f.get("FaceOccluded", {}).get("Value") or f.get("Sunglasses", {}).get("Value")),
                    "yaw": round(f["Pose"]["Yaw"]), "box": {k: round(v, 4) for k, v in b.items()}})
    return out


def compare(rek, source_crop, target_crop):
    try:
        r = rek.compare_faces(SourceImage={"Bytes": source_crop}, TargetImage={"Bytes": target_crop},
                              SimilarityThreshold=0, QualityFilter="NONE")
    except rek.exceptions.InvalidParameterException:
        return 0.0  # no usable face in one of the crops
    return max([m["Similarity"] for m in r["FaceMatches"]], default=0.0)


def reference_faces(rek, datas):
    """All faces from all reference images, tagged with their image index."""
    with ThreadPoolExecutor(4) as ex:
        per = list(ex.map(lambda d: faces(rek, d), datas))
    refs = [{**f, "ref": i} for i, fs in enumerate(per) for f in fs][:MAX_REFERENCE_FACES]
    return refs, [len(fs) for fs in per]


def verdict(score, face, pass_at=PASS_AT, review_at=REVIEW_AT, min_face=MIN_FACE):
    if face["face_px"] < min_face:
        return "REVIEW", f"face too small to judge ({face['face_px']} px tall)"
    if face["occluded"]:
        return "REVIEW", "face partly covered"
    if score >= pass_at:
        return "PASS", ""
    if score < review_at:
        return "FAIL", ""
    return "REVIEW", f"close call: between {review_at:g} and {pass_at:g}"


def score_image(rek, refs, data, pass_at=PASS_AT, review_at=REVIEW_AT, min_face=MIN_FACE, workers=8):
    """Best match of this image's faces against the reference faces, with its verdict."""
    targets = faces(rek, data)[:MAX_FACES]
    if not targets:
        return {"verdict": "REVIEW", "score": None, "reason": "no face found", "faces_found": 0, "clear": False}
    pairs = [(t, r) for t in range(len(targets)) for r in range(len(refs))]
    with ThreadPoolExecutor(workers) as ex:
        sims = list(ex.map(lambda p: compare(rek, refs[p[1]]["crop"], targets[p[0]]["crop"]), pairs))
    (t, r), sim = max(zip(pairs, sims), key=lambda x: x[1])
    face = targets[t]
    v, why = verdict(sim, face, pass_at, review_at, min_face)
    return {"verdict": v, "score": round(sim, 2), "reason": why, "face_px": face["face_px"],
            "clear": face["face_px"] >= min_face and not face["occluded"],
            "occluded": face["occluded"], "face_box": face["box"], "faces_found": len(targets),
            "matched_reference": refs[r]["ref"]}


def aggregate_frames(frames, pass_at=PASS_AT, review_at=REVIEW_AT):
    """Video verdict from per-frame results [{t, verdict, score, reason, ...}].

    A clear frame under review_at means the character changed somewhere in the clip (a different face or a hidden
    cut), so it fails even if other frames match. PASS needs every clear frame at pass_at or above."""
    clear = [f for f in frames if f.get("clear")]
    if not clear:
        return {"verdict": "REVIEW", "reason": "no clear face in the sampled frames"}
    worst = min(clear, key=lambda f: f["score"])
    if worst["score"] < review_at:
        return {"verdict": "FAIL", "reason": f"face does not match at {worst['t']:.1f}s (score {worst['score']:.1f})"}
    if all(f["score"] >= pass_at for f in clear):
        return {"verdict": "PASS", "reason": f"{len(clear)} clear frames, lowest {worst['score']:.1f}"}
    return {"verdict": "REVIEW", "reason": f"close call at {worst['t']:.1f}s (score {worst['score']:.1f})"}
