"""Blind pairwise LLM judge (Amazon Bedrock Converse) between two bench runs.

  python bench/judge.py --suite bench/suite.json --a out/bench/r0 --b out/bench/r1 --out out/bench/judge_r0_vs_r1.json

For every video case present in both runs: 6 evenly spaced frames per clip go to the model with the prompt,
twice with the clip order swapped. A win counts only if both orders agree, otherwise it is a tie.
Answers are cached under <out dir>/.judge_cache by (case, prompt, model, rubric, sha of both files): re-runs are free.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import media  # noqa: E402

CRITERIA = ["prompt_adherence", "visual_detail", "motion", "anatomy_artifacts", "temporal_consistency"]
N_FRAMES = 6
RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "InternalServerException",
             "TooManyRequestsException", "ModelErrorException"}

SYSTEM = "You are a strict, impartial judge of AI-generated video quality. You compare two clips rendered from the same request and answer only with JSON."

RUBRIC = """Both clips were generated from this request:
<prompt>{prompt}</prompt>

Each clip is shown as {n} frames sampled evenly from its first to its last frame, in order (times in seconds: {times}).
Judge the clips as videos, using the frame sequence to infer motion. Criteria:
- prompt_adherence: subject, action, setting, camera and style match the request.
- visual_detail: sharpness, texture, lighting quality; no mush, smearing or banding.
- motion: enough movement for what the prompt asks AND movement that looks physically natural (not frozen, not slow-motion when real-time is implied, not warping).
- anatomy_artifacts: hands, faces, bodies, animals and objects are well-formed; no extra limbs, melting, garbled text or duplicated objects. The better clip has FEWER artifacts.
- temporal_consistency: identity, objects, layout and lighting stay stable from frame to frame; no flicker, popping or morphing.

For each criterion answer "1", "2" or "tie" (tie only if you genuinely cannot separate them), then an overall winner,
your confidence 1-5 in the overall call, and one sentence giving the main reason (refer to the clips as Clip 1 and Clip 2).
Reply with only this JSON object:
{{"prompt_adherence": "1|2|tie", "visual_detail": "1|2|tie", "motion": "1|2|tie", "anatomy_artifacts": "1|2|tie", "temporal_consistency": "1|2|tie", "overall": "1|2|tie", "confidence": 1-5, "reason": "..."}}"""
RUBRIC_SHA = hashlib.sha256((SYSTEM + RUBRIC).encode()).hexdigest()[:12]


class Judge:
    def __init__(self, model, profile, region, cache_dir):
        session = boto3.Session(profile_name=profile, region_name=region)
        self.brt = session.client("bedrock-runtime", config=Config(retries={"max_attempts": 6, "mode": "adaptive"},
                                                                   read_timeout=300, max_pool_connections=16))
        self.model, self.cache_dir = model, cache_dir
        self._frames, self._sha, self._lock = {}, {}, threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)

    def sha(self, path):
        with self._lock:
            if path not in self._sha:
                self._sha[path] = media.sha256(path)
            return self._sha[path]

    def frames(self, path):
        """N_FRAMES JPEGs from the first to the last frame, plus their timestamps."""
        with self._lock:
            if path in self._frames:
                return self._frames[path]
        info = media.probe(path)
        fps = info["fps"] or 24.0
        n = info["frames"] or max(1, int(round((info["duration"] or 0) * fps)))
        idx = [round(i * (n - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)]
        times = [i / fps for i in idx]
        # a quarter frame early so the seek lands on frame idx even with rounding in the container timestamps
        jpgs = [media.jpeg_frame(path, t - 0.25 / fps) for t in times]
        if not all(jpgs):
            raise RuntimeError(f"frame extraction failed for {path}")
        with self._lock:
            self._frames[path] = (jpgs, times)
        return jpgs, times

    def compare(self, case_id, prompt, clip1, clip2):
        """Raw verdict for one order (clip1 shown first). Cached on disk."""
        key = hashlib.sha256(json.dumps([case_id, prompt, self.model, RUBRIC_SHA, self.sha(clip1), self.sha(clip2)]).encode()).hexdigest()
        cpath = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cpath):
            return json.load(open(cpath))
        f1, t1 = self.frames(clip1)
        f2, _ = self.frames(clip2)
        text = RUBRIC.format(prompt=prompt, n=N_FRAMES, times=", ".join(f"{t:.2f}" for t in t1))
        content = [{"text": text}, {"text": "Clip 1:"}, *[{"image": {"format": "jpeg", "source": {"bytes": b}}} for b in f1],
                   {"text": "Clip 2:"}, *[{"image": {"format": "jpeg", "source": {"bytes": b}}} for b in f2],
                   {"text": "Return the JSON object now."}]
        verdict = None
        for attempt in range(6):
            try:
                resp = self.brt.converse(modelId=self.model, system=[{"text": SYSTEM}],
                                         messages=[{"role": "user", "content": content}], inferenceConfig={"maxTokens": 1500})
            except ClientError as exc:
                if exc.response["Error"]["Code"] in RETRYABLE and attempt < 5:
                    time.sleep(min(90, 10 * 2 ** attempt) * random.uniform(0.5, 1.0))
                    continue
                raise
            out = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
            verdict = parse(out)
            if verdict:
                break
            print(f"  {case_id}: unparseable answer, retrying: {out[:200]!r}", file=sys.stderr)
        if not verdict:
            raise RuntimeError(f"{case_id}: no valid JSON verdict")
        verdict["usage"] = resp.get("usage")
        json.dump(verdict, open(cpath, "w"), indent=1)
        return verdict


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    pick = {k: str(v.get(k, "")).strip().lower().removeprefix("clip ") for k in CRITERIA + ["overall"]}
    if not all(p in ("1", "2", "tie") for p in pick.values()):
        return None
    return {**pick, "confidence": v.get("confidence"), "reason": str(v.get("reason", ""))}


def combine(ab, ba, name_a, name_b):
    """Merge the two orders into a/b/tie per criterion; a side wins only if both orders pick it."""
    side_ab = {"1": "a", "2": "b", "tie": "tie"}
    side_ba = {"1": "b", "2": "a", "tie": "tie"}
    crit = {}
    for k in CRITERIA + ["overall"]:
        x, y = side_ab[ab[k]], side_ba[ba[k]]
        crit[k] = x if x == y else "tie"

    def named(reason, first, second):
        reason = re.sub(r"\b[Cc]lip 1\b", first, reason)
        return re.sub(r"\b[Cc]lip 2\b", second, reason)
    return {"winner": crit.pop("overall"), "criteria": crit, "confidence": [ab.get("confidence"), ba.get("confidence")],
            "agreed": side_ab[ab["overall"]] == side_ba[ba["overall"]],
            "reason": named(ab["reason"], name_a, name_b), "reason_swapped": named(ba["reason"], name_b, name_a)}


def summarize(cases):
    out = {}
    for k in ["overall"] + CRITERIA:
        picks = [c["winner"] if k == "overall" else c["criteria"][k] for c in cases.values() if "winner" in c]
        out[k] = {"win": picks.count("b"), "tie": picks.count("tie"), "loss": picks.count("a")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--a", required=True, help="baseline run dir")
    ap.add_argument("--b", required=True, help="candidate run dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="us.anthropic.claude-opus-4-7")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "tycenj-prod"))
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", help="comma-separated case ids")
    a = ap.parse_args()

    suite = media.load_suite(a.suite)
    ca, cb = media.clips(a.a), media.clips(a.b)
    ids = [i for i in ca if i in cb and ca[i].endswith(".mp4") and (not a.only or i in a.only.split(","))]
    missing = [i for i in ids if i not in suite]
    if missing:
        print(f"skipping cases not in the suite (no prompt): {missing}", file=sys.stderr)
        ids = [i for i in ids if i in suite]
    name_a, name_b = os.path.basename(os.path.normpath(a.a)), os.path.basename(os.path.normpath(a.b))
    if name_a == name_b:
        name_a, name_b = "A", "B"

    judge = Judge(a.model, a.profile, a.region, os.path.join(os.path.dirname(os.path.abspath(a.out)), ".judge_cache"))
    raw = {}
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(judge.compare, cid, suite[cid]["prompt"], *pair): (cid, order)
                for cid in ids for order, pair in (("ab", (ca[cid], cb[cid])), ("ba", (cb[cid], ca[cid])))}
        for fut in as_completed(futs):
            cid, order = futs[fut]
            try:
                raw.setdefault(cid, {})[order] = fut.result()
            except Exception as exc:  # one bad case shouldn't sink the run; it is reported and left out of the summary
                raw.setdefault(cid, {})["error"] = f"{order}: {type(exc).__name__}: {exc}"
                print(f"{cid} ({order}) failed: {exc}", file=sys.stderr)

    cases = {}
    for cid in ids:
        r = raw.get(cid, {})
        if "error" in r or len(r) < 2:
            cases[cid] = {"error": r.get("error", "incomplete")}
            continue
        cases[cid] = {**combine(r["ab"], r["ba"], name_a, name_b), "orders": r}
        c = cases[cid]
        print(f"{cid}: {dict(a=name_a, b=name_b).get(c['winner'], 'tie')} "
              f"({'agreed' if c['agreed'] else 'orders disagree'}, conf {c['confidence']}) {c['reason']}")

    summary = summarize(cases)
    json.dump({"a": a.a, "b": a.b, "name_a": name_a, "name_b": name_b, "model": a.model, "rubric": RUBRIC_SHA,
               "criteria": CRITERIA, "summary": summary, "cases": cases}, open(a.out, "w"), indent=1)
    print(f"\n{name_b} vs {name_a} (win/tie/loss for {name_b}), {sum('winner' in c for c in cases.values())} cases:")
    for k, v in summary.items():
        print(f"  {k:22s} {v['win']}/{v['tie']}/{v['loss']}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
