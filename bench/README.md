# bench: recipe A/B harness

Render a fixed suite with each recipe, measure every clip, have an LLM judge pairs blind, then compare everything on one page.

| script | what it does |
|---|---|
| `run.py` | Turns `suite.json` into generate_video / generate_audio jobs (`recipe`, optional `tuning`, `idempotency_key = bench:<name>:<case_id>`), uploads every local input file, and runs them with `examples/batch.py`. Writes `out/bench/<name>/<case_id>.mp4` (`.wav` for audio), `jobs.json`, `results.json`. |
| `metrics.py` | Writes `metrics.json`: size, fps, frames, duration; motion (Farneback flow at 320 px wide, px/s); sharpness (Laplacian variance at 720 px high, 12 frames); flicker (mean abs change of frame luma); audio rate, channels, bitrate, LUFS, true peak, LRA; WER against `expect_speech` (faster-whisper `small.en`, CPU). |
| `judge.py` | Bedrock Converse judge (default `us.anthropic.claude-opus-4-7`): 6 frames per clip plus the prompt, per-criterion and overall winner. Each pair runs in both orders and a win counts only if both agree. Answers are cached by file hash, so re-runs cost nothing. |
| `compare.py` | Builds a self-contained HTML page with a summary table (cost, render time, metric means and deltas, judge W/T/L against the first run) and one row per case with synced playback, sound solo, and picks you can export as JSON. |

```bash
python3.12 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt   # or: uv venv / uv pip
export GATEWAY_URL=... AWS_PROFILE=tycenj-prod                                           # run.py only

bench/.venv/bin/python bench/run.py bench/suite.json --recipe r1 --name r1 [--tuning '{"steps": 30}'] [--only a,b] [--dry-run]
bench/.venv/bin/python bench/metrics.py out/bench/r1/ --suite bench/suite.json
bench/.venv/bin/python bench/judge.py --suite bench/suite.json --a out/bench/r0 --b out/bench/r1 --out out/bench/judge_r0_vs_r1.json
bench/.venv/bin/python bench/compare.py --suite bench/suite.json out/bench/r0 out/bench/r1 --judge out/bench/judge_*.json --out out/bench/compare.html
```

Suite format: see `suite.sample.json`. A case holds any generate_video arguments plus `id` and `expect_speech` (the spoken or sung line, or `null`). Local paths in `first_frame`, `last_frame`, `refs[]`, `keyframes[].key`, `soundtrack.key`, `ref_audios[]` and `ref_videos[]` resolve against the working directory first, then the suite's own folder. Cases on an audio backend (`stable-audio-3`) go to generate_audio.

Notes:
- Idempotency keys never expire. To re-render a case that FAILED, use a new `--name`.
- The judge only sees frames, so it never hears audio. Audio quality comes from LUFS and WER in metrics.
- Use the bench venv rather than Homebrew's Python: the pin `numpy<2.3` is deliberate.
