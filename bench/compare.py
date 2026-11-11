"""One self-contained HTML page comparing bench runs side by side (the first run is the baseline).

  python bench/compare.py --suite bench/suite.json out/bench/r0 out/bench/r1 [more runs...] \
      --judge out/bench/judge_*.json --out out/bench/compare.html

Clips are linked by relative path (keep the page next to the run dirs); no external assets.
"""
import argparse
import glob
import html
import json
import os
import statistics
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import media  # noqa: E402

# key, label, unit, format, which direction is better (None = neither), getter(result, metrics)
STATS = [
    ("cost", "cost", "$", "{:.4f}", "low", lambda r, m: (r.get("cost") or {}).get("usd")),
    ("render", "render", "s", "{:.1f}", "low", lambda r, m: (r.get("timings") or {}).get("render_s")),
    ("motion", "motion", "px/s", "{:.1f}", None, lambda r, m: m.get("motion")),
    ("sharpness", "sharpness", "", "{:.0f}", "high", lambda r, m: m.get("sharpness")),
    ("flicker", "flicker", "", "{:.3f}", "low", lambda r, m: m.get("flicker")),
    ("lufs", "LUFS", "", "{:.1f}", None, lambda r, m: (m.get("audio") or {}).get("lufs")),
    ("wer", "WER", "", "{:.2f}", "low", lambda r, m: (m.get("speech") or {}).get("wer")),
]
CLIP_STATS = ("render", "cost", "motion", "sharpness", "lufs", "wer")
ABS_DELTA = {"lufs": " LU", "wer": ""}  # log / ratio scales: an absolute delta reads better than a percentage
QUALITY_ORDER = {"full": 0, "draft8": 1, "draft4": 2}
CRITERIA = ["prompt_adherence", "visual_detail", "motion", "anatomy_artifacts", "temporal_consistency"]
e = html.escape


def load_run(d):
    jobs = media.load_json(os.path.join(d, "jobs.json"), [])
    return {"dir": d, "real": os.path.realpath(d), "name": os.path.basename(os.path.normpath(d)), "clips": media.clips(d),
            "results": media.load_json(os.path.join(d, "results.json"), {}),
            "metrics": media.load_json(os.path.join(d, "metrics.json"), {}),
            "recipe": sorted({str(j.get("recipe")) for j in jobs if j.get("recipe")}),
            "tuning": sorted({json.dumps(j["tuning"], sort_keys=True) for j in jobs if j.get("tuning")})}


def stat(run, key, cid):
    getter = next(s[5] for s in STATS if s[0] == key)
    r = run["results"].get(cid, {})
    if key in ("cost", "render") and r.get("status") not in (None, "SUCCEEDED"):
        return None
    v = getter(r, run["metrics"].get(cid, {}))
    return v if isinstance(v, (int, float)) else None


def fmt(key, v):
    if v is None:
        return "-"
    _, _, unit, f, _, _ = next(s for s in STATS if s[0] == key)
    return ("$" + f.format(v)) if unit == "$" else f.format(v) + (f" {unit}" if unit else "")


def verdicts(judge_paths, runs):
    """{run index: {case: {result: win|tie|loss, ...}}} for every run judged against the first run."""
    base = runs[0]["real"]
    out = {}
    for p in judge_paths:
        j = json.load(open(p))
        a, b = os.path.realpath(j["a"]), os.path.realpath(j["b"])
        for i, run in enumerate(runs[1:], 1):
            if {a, b} != {base, run["real"]}:
                continue
            mine = "b" if b == run["real"] else "a"
            to = {mine: "win", "tie": "tie"}
            for cid, c in j["cases"].items():
                if "winner" not in c:
                    out.setdefault(i, {})[cid] = {"result": "error", "reason": c.get("error", "")}
                    continue
                conf = c.get("confidence") or []
                out.setdefault(i, {})[cid] = {
                    "result": to.get(c["winner"], "loss"), "reason": c.get("reason", ""),
                    "confidence": "/".join(str(x) for x in conf if x is not None),
                    "criteria": {k: to.get(v, "loss") for k, v in c.get("criteria", {}).items()}, "model": j.get("model")}
    return out


def summary_table(runs, case_ids, judged):
    head = "".join(f"<th>{e(s[1])}{(' ' + e(s[2])) if s[2] and s[2] != '$' else ''}</th>" for s in STATS)
    rows = []
    for i, run in enumerate(runs):
        ok = sum(run["results"].get(c, {}).get("status") == "SUCCEEDED" or c in run["clips"] for c in case_ids)
        cells = []
        for key, _, _, _, better, _ in STATS:
            # mean over the cases every run has a value for, so the deltas compare like with like
            common = [c for c in case_ids if all(stat(r, key, c) is not None for r in runs)]
            if not common:
                cells.append("<td class=na>-</td>")
                continue
            mean = statistics.fmean(stat(run, key, c) for c in common)
            base = statistics.fmean(stat(runs[0], key, c) for c in common)
            delta, cls = "", ""
            if i and key in ABS_DELTA:
                d = mean - base
                delta = f"{d:+.2f}{ABS_DELTA[key]}"
            elif i and base:
                d = (mean - base) / abs(base) * 100
                delta = f"{d:+.0f}%"
            if delta and better and abs(mean - base) > 1e-9:
                cls = "good" if (mean < base) == (better == "low") else "bad"
            cells.append(f"<td title='mean over {len(common)} cases'>{e(fmt(key, mean))}"
                         f"<div class='delta {cls}'>{e(delta) if i else 'baseline'}</div></td>")
        wtl = ""
        if i and judged.get(i):
            res = [v["result"] for v in judged[i].values()]
            wtl = f"{res.count('win')}/{res.count('tie')}/{res.count('loss')}"
        recipe = ", ".join(run["recipe"]) or "-"
        tuning = "; ".join(run["tuning"])
        rows.append(f"<tr><th class=run>{e(run['name'])}</th><td>{e(recipe)}"
                    f"{f'<div class=delta>{e(tuning)}</div>' if tuning else ''}</td>"
                    f"<td>{ok}/{len(case_ids)}</td>{''.join(cells)}<td>{e(wtl) if i else '-'}</td></tr>")
    return (f"<table class=summary><tr><th>run</th><th>recipe</th><th>clips</th>{head}<th>judge W/T/L<br>"
            f"<span class=delta>vs {e(runs[0]['name'])}</span></th></tr>{''.join(rows)}</table>")


def clip_cell(i, run, cid, judged, out_dir):
    path = run["clips"].get(cid)
    res = run["results"].get(cid, {})
    pick = f"<label class=pick><input type=checkbox class=pick data-case='{e(cid)}' data-run='{e(run['name'])}'> pick</label>"
    if path:
        src = urllib.parse.quote(os.path.relpath(path, out_dir))
        if path.endswith(".wav"):
            player = f"<div class=audiobox>audio<audio src='{src}' preload=metadata muted data-i={i}></audio></div>"
        else:
            player = f"<video src='{src}' preload=metadata muted playsinline data-i={i}></video>"
    else:
        status = res.get("status") or "not rendered"
        player = f"<div class=missing><b>{e(status)}</b><span>{e(str(res.get('error') or '')[:300])}</span></div>"
    lines = []
    for key in CLIP_STATS if path else ():
        v = stat(run, key, cid)
        if v is None and key not in ("render", "cost"):
            continue
        title = ""
        if key == "wer" and v is not None:
            title = f" title='{e(run['metrics'][cid]['speech'].get('transcript', ''))}'"
        label = next(s[1] for s in STATS if s[0] == key)
        lines.append(f"<li{title}><span>{e(label)}</span>{e(fmt(key, v))}</li>")
    v = judged.get(i, {}).get(cid)
    if v:
        crit = ", ".join(f"{k} {r}" for k, r in v.get("criteria", {}).items())
        conf = f" ({v['confidence']})" if v.get("confidence") else ""
        lines.append(f"<li class='judge {v['result']}' title='{e(crit)}'><span>judge</span>{e(v['result'])}{e(conf)}</li>")
        if v.get("reason"):
            lines.append(f"<li class=reason>{e(v['reason'])}</li>")
    return (f"<figure class=clip><figcaption><b>{e(run['name'])}</b>{pick}</figcaption>{player}"
            f"<ul class=stats>{''.join(lines)}</ul></figure>")


def page(runs, suite, judged, out_dir):
    ids = list(suite) + sorted({c for r in runs for c in list(r["clips"]) + list(r["results"]) if c not in suite})
    ids = [c for c in ids if any(c in r["clips"] or c in r["results"] for r in runs)]
    groups = {}
    for cid in ids:
        case = suite.get(cid, {})
        groups.setdefault((case.get("backend") or "(not in suite)", case.get("quality") or "-"), []).append(cid)
    body = []
    for (backend, quality) in sorted(groups, key=lambda g: (g[0], QUALITY_ORDER.get(g[1], 9), g[1])):
        body.append(f"<h2>{e(backend)} <span>{e(quality) if quality != '-' else ''}</span></h2>")
        for cid in groups[(backend, quality)]:
            prompt = suite.get(cid, {}).get("prompt", "")
            sound = "".join(f"<button data-s='{i}'>{e(r['name'])}</button>" for i, r in enumerate(runs))
            cells = "".join(clip_cell(i, r, cid, judged, out_dir) for i, r in enumerate(runs))
            body.append(f"""<section class=row>
<div class=head><div class=title><b>{e(cid)}</b><span class=prompt title="{e(prompt)}">{e(prompt)}</span></div>
<div class=ctl><button class=play>play</button><input class=scrub type=range min=0 max=1 step=0.01 value=0><span class=t>0.00 s</span>
<span class=snd>sound <button data-s=off class=on>off</button>{sound}<button data-s=all>all</button></span></div></div>
<div class=clips style="--n:{len(runs)}">{cells}</div></section>""")
    names = json.dumps([r["name"] for r in runs]).replace("</", "<\\/")
    model = next((v["model"] for d in judged.values() for v in d.values() if v.get("model")), None)
    note = f"judge: {e(model)}, blind, both orders must agree. " if model else ""
    return f"""<!doctype html><html><head><meta charset=utf-8><title>bench: {e(' vs '.join(r['name'] for r in runs))}</title>
<style>{CSS}</style></head><body>
<header><h1>bench <span>{e(' / '.join(r['name'] for r in runs))}</span></h1><button id=export>export picks</button><span id=npicks></span></header>
{summary_table(runs, ids, judged)}
<p class=note>Means over cases with a value in every run; deltas vs {e(runs[0]['name'])}. {note}Hover a WER for the transcript, a verdict for per-criterion results.</p>
{''.join(body)}
<script>const RUNS = {names};{JS}</script></body></html>"""


CSS = """
:root{--bg:#0e1014;--panel:#161a21;--line:#262c36;--fg:#e4e7ec;--mute:#8b93a1;--acc:#7cb0ff;--good:#5ccf8a;--bad:#ff7b7b}
*{box-sizing:border-box}body{margin:0;padding:24px 32px 80px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif;font-variant-numeric:tabular-nums}
header{display:flex;align-items:center;gap:14px;margin-bottom:18px}h1{font-size:20px;margin:0;flex:1}h1 span,h2 span{color:var(--mute);font-weight:400}
h2{font-size:16px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
button{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);color:#0b0d10;border-color:var(--acc)}
table.summary{border-collapse:collapse;width:100%;background:var(--panel);border-radius:8px;overflow:hidden}
.summary th,.summary td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
.summary th{color:var(--mute);font-weight:500;font-size:12px}.summary th.run{color:var(--fg);font-size:14px}.summary th:first-child,.summary td:nth-child(2),.summary tr:first-child th:nth-child(2){text-align:left}
.summary tr:last-child td,.summary tr:last-child th{border-bottom:0}.na{color:var(--mute)}
.delta{font-size:11px;color:var(--mute)}.delta.good{color:var(--good)}.delta.bad{color:var(--bad)}
.note{color:var(--mute);font-size:12px;margin:8px 0 0}
.row{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.head{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;margin-bottom:10px}
.title{flex:1 1 400px;min-width:0;display:flex;gap:10px;align-items:baseline}
.prompt{color:var(--mute);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.ctl{display:flex;gap:8px;align-items:center}.scrub{width:220px;accent-color:var(--acc)}.t{color:var(--mute);width:56px}
.snd{color:var(--mute);display:flex;gap:4px;align-items:center;margin-left:8px}.snd button{padding:2px 8px;font-size:12px}
.clips{display:grid;grid-template-columns:repeat(var(--n),minmax(0,1fr));gap:14px}
.clip{margin:0;min-width:0}figcaption{display:flex;justify-content:space-between;margin-bottom:6px}
.pick{color:var(--mute);font-size:12px;cursor:pointer}.pick input{accent-color:var(--good);vertical-align:-2px}
video{display:block;width:100%;max-height:68vh;object-fit:contain;background:#000;border-radius:6px;cursor:pointer}
.audiobox,.missing{display:flex;align-items:center;justify-content:center;height:90px;background:#000;border-radius:6px;color:var(--mute);text-align:center;padding:8px}
.missing{flex-direction:column;gap:4px}.missing b{color:var(--bad)}
.stats{list-style:none;margin:8px 0 0;padding:0;font-size:13px;max-width:380px}.stats li{display:flex;justify-content:space-between;gap:8px;padding:1px 0}
.stats li span{color:var(--mute)}.stats li.reason{display:block;color:var(--mute);font-size:12px;line-height:1.35;padding-top:3px}
.judge.win{color:var(--good)}.judge.loss{color:var(--bad)}
"""

JS = r"""
const fmtT = t => (isFinite(t) ? t : 0).toFixed(2) + ' s';
document.querySelectorAll('.row').forEach(row => {
  const media = [...row.querySelectorAll('video, audio')];
  const ctl = row.querySelector('.ctl'), play = row.querySelector('.play'), scrub = row.querySelector('.scrub'), time = row.querySelector('.t');
  if (!media.length) { ctl.style.visibility = 'hidden'; return; }
  let master = media[0], timer = null;
  const longest = () => media.reduce((a, b) => (b.duration || 0) > (a.duration || 0) ? b : a);
  const seek = t => media.forEach(m => { if (m.readyState) m.currentTime = Math.min(t, Math.max(0, (m.duration || t) - 0.01)); });
  // the longest clip is the clock; shorter ones are nudged back when they drift and simply wait at their end
  const sync = () => media.forEach(m => {
    if (m !== master && !m.paused && master.currentTime < (m.duration || 0) - 0.05 && Math.abs(m.currentTime - master.currentTime) > 0.08) m.currentTime = master.currentTime;
  });
  const start = () => { seek(master.ended ? 0 : master.currentTime); media.forEach(m => m.play().catch(() => {})); play.textContent = 'pause'; timer = setInterval(sync, 250); };
  const stop = () => { media.forEach(m => m.pause()); play.textContent = 'play'; clearInterval(timer); timer = null; };
  play.onclick = () => timer ? stop() : start();
  scrub.oninput = () => { seek(+scrub.value); time.textContent = fmtT(+scrub.value); if (timer) media.forEach(m => m.play().catch(() => {})); };
  media.forEach(m => {
    m.addEventListener('loadedmetadata', () => { master = longest(); scrub.max = master.duration || 1; });
    m.addEventListener('timeupdate', () => { if (m === master) { scrub.value = m.currentTime; time.textContent = fmtT(m.currentTime); } });
    m.addEventListener('ended', () => { if (m === master && timer) { seek(0); media.forEach(x => x.play().catch(() => {})); } });
    m.addEventListener('click', () => play.click());
  });
  row.querySelectorAll('.snd button').forEach(b => b.onclick = () => {
    const s = b.dataset.s;
    media.forEach(m => { m.muted = !(s === 'all' || s === m.dataset.i); });
    row.querySelectorAll('.snd button').forEach(x => x.classList.toggle('on', x === b));
  });
});
const KEY = 'bench-picks:' + RUNS.join('|');
const picks = JSON.parse(localStorage.getItem(KEY) || '{}');
const count = () => { document.getElementById('npicks').textContent = Object.keys(picks).length + ' picked'; };
document.querySelectorAll('input.pick').forEach(cb => {
  const k = cb.dataset.case + '|' + cb.dataset.run;
  cb.checked = !!picks[k];
  cb.onchange = () => { if (cb.checked) picks[k] = 1; else delete picks[k]; localStorage.setItem(KEY, JSON.stringify(picks)); count(); };
});
count();
document.getElementById('export').onclick = () => {
  const out = {};
  Object.keys(picks).sort().forEach(k => { const i = k.lastIndexOf('|'); (out[k.slice(0, i)] ||= []).push(k.slice(i + 1)); });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({runs: RUNS, picks: out}, null, 1)], {type: 'application/json'}));
  a.download = 'picks.json';
  a.click();
};
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs; the first is the baseline")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--judge", nargs="*", default=[], help="judge JSON files (globs ok)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    runs = [load_run(d) for d in a.runs]
    judge_paths = sorted({p for g in a.judge for p in (glob.glob(g) or [g]) if os.path.exists(p)})
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    judged = verdicts(judge_paths, runs)
    with open(a.out, "w") as fh:
        fh.write(page(runs, media.load_suite(a.suite), judged, out_dir))
    print(f"{len(runs)} runs, {len(judge_paths)} judge files -> {a.out}")


if __name__ == "__main__":
    main()
