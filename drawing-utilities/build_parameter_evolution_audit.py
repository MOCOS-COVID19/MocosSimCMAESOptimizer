#!/usr/bin/env python3
"""Build a self-contained interactive parameter-evolution audit.

The history file supplies candidate scores and identities; parameter values are
read from the corresponding ``real_sims/<stage>/iter_N/cand_NN/config.json``.
"""

import argparse
import json
import math
from pathlib import Path


STAGES = (6, 9, 12)
SCALARS = ("school", "class", "age_coupling_param")
VECTORS = (
    "infection_modulation",
    "mild_detection_modulation",
    "tracing_modulation",
)


def nested(obj, *path):
    for key in path:
        obj = obj[key]
    return obj


def candidate_config_path(config_root: Path, entry: dict) -> Path:
    return (config_root / entry["stage"] / f"iter_{int(entry['iteration'])}" /
            f"cand_{int(entry['candidate']):02d}" / "config.json")


def build_records(history: list, config_root: Path) -> list:
    records = []
    for entry in history:
        stage = int(entry.get("fit_months", 0))
        if stage not in STAGES or entry.get("status") not in ("ok", "completed"):
            continue
        path = candidate_config_path(config_root, entry)
        if not path.exists():
            raise FileNotFoundError(f"candidate config not found: {path}")
        config = json.loads(path.read_text())
        transmission = config.get("transmission_probabilities",
                                  config.get("transmission_probability"))
        if not isinstance(transmission, dict):
            raise KeyError(f"transmission probabilities missing from {path}")
        try:
            score = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        records.append({
            "stage": stage,
            "iteration": int(entry["iteration"]),
            "candidate": int(entry["candidate"]),
            "score": score,
            "scalars": {name: float(transmission[name]) for name in SCALARS},
            "vectors": {
                name: [float(value) for value in nested(
                    config, name, "params", "interval_values")]
                for name in VECTORS
            },
        })
    return records


def build_handoff_audit(records: list) -> list:
    """Compare each predecessor winner with the next stage's locked prefix."""
    result = []
    for source_stage, target_stage in ((6, 9), (9, 12)):
        source = [record for record in records if record["stage"] == source_stage]
        target = [record for record in records if record["stage"] == target_stage]
        if not source or not target:
            continue
        winner = min(source, key=lambda record: record["score"])
        first = min(target, key=lambda record: (record["iteration"], record["candidate"]))
        vectors = {}
        for name in VECTORS:
            expected = winner["vectors"][name][:source_stage]
            actual = first["vectors"][name][:source_stage]
            vectors[name] = {
                "matches": expected == actual,
                "different_buckets": [index + 1 for index, (left, right) in
                                      enumerate(zip(expected, actual)) if left != right],
            }
        result.append({"source_stage": source_stage, "target_stage": target_stage,
                       "source_winner": [winner["iteration"], winner["candidate"]],
                       "target_reference": [first["iteration"], first["candidate"]],
                       "vectors": vectors})
    return result


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parameter evolution audit</title><style>
:root{color-scheme:dark;--bg:#08111f;--panel:#111d30;--line:#2a3c59;--text:#edf3ff;--muted:#9eb0ca;--accent:#63b3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}header,main{max-width:1450px;margin:auto;padding:24px}h1{margin:0 0 8px}.note{color:var(--muted);line-height:1.5}.controls,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}.controls{display:flex;gap:24px;flex-wrap:wrap}fieldset{border:0;padding:0;margin:0;display:flex;gap:12px;align-items:center}legend{color:var(--muted);font-size:12px;margin-bottom:8px}label{cursor:pointer}select{background:#0b1627;color:var(--text);border:1px solid var(--line);padding:7px;border-radius:6px}canvas{display:block;width:100%;height:360px;background:#0b1627;border-radius:8px}.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;color:var(--muted)}.swatch{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:50%}@media(max-width:600px){header,main{padding:14px}}
</style></head><body><header><h1>Parameter evolution audit</h1><div class="note">Candidate configurations joined to <code>optimizer_history.json</code>. Points are ordered by iteration and candidate; lower score is better.</div></header><main>
<section class="controls"><fieldset><legend>Stages</legend><label><input type="checkbox" data-stage="6" checked> 6m</label><label><input type="checkbox" data-stage="9" checked> 9m</label><label><input type="checkbox" data-stage="12" checked> 12m</label></fieldset>
<fieldset><legend>Scalar series</legend><label><input type="checkbox" data-scalar="school" checked> school</label><label><input type="checkbox" data-scalar="class" checked> class</label><label><input type="checkbox" data-scalar="age_coupling_param" checked> age coupling</label></fieldset>
<fieldset><legend>Vector</legend><select id="vector"><option value="infection_modulation">Infection modulation</option><option value="mild_detection_modulation">Mild-detection modulation</option><option value="tracing_modulation">Tracing modulation</option></select></fieldset></section>
<section class="panel"><h2>Stage handoff integrity</h2><div id="handoff"></div><div class="note">A failing row means the next stage did not lock the preceding stage winner's fitted prefix.</div></section>
<section class="panel"><h2>Scalar evolution</h2><canvas id="scalar"></canvas><div id="scalarLegend" class="legend"></div></section>
<section class="panel"><h2 id="vectorTitle">Vector evolution</h2><canvas id="vectorPlot"></canvas><div class="note">Each line is one interval bucket. Stage changes are shown as gaps because horizons have different active vector lengths.</div></section>
</main><script>const DATA=__DATA__,HANDOFF=__HANDOFF__;const colors={school:'#66c2ff',class:'#ffad5c',age_coupling_param:'#c084fc'};
const selectedStages=()=>new Set([...document.querySelectorAll('[data-stage]:checked')].map(x=>+x.dataset.stage));
function size(c){const d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);return{x,w,h}}
function axes(c,lo,hi){const {x,w,h}=size(c),p={l:55,r:18,t:18,b:36};x.strokeStyle='#2a3c59';x.fillStyle='#9eb0ca';x.font='11px system-ui';for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,y=h-p.b-(h-p.t-p.b)*i/4;x.beginPath();x.moveTo(p.l,y);x.lineTo(w-p.r,y);x.stroke();x.fillText(v.toFixed(2),5,y+4)}return{x,w,h,p,X:i=>p.l+(w-p.l-p.r)*i/Math.max(1,DATA.length-1),Y:v=>h-p.b-(h-p.t-p.b)*(v-lo)/(hi-lo)}}
function drawScalars(){const enabled=new Set([...document.querySelectorAll('[data-scalar]:checked')].map(x=>x.dataset.scalar)),st=selectedStages(),a=axes(document.querySelector('#scalar'),0,1);Object.keys(colors).filter(k=>enabled.has(k)).forEach(k=>{a.x.strokeStyle=colors[k];a.x.lineWidth=1.5;a.x.beginPath();let pen=false;DATA.forEach((r,i)=>{if(!st.has(r.stage)){pen=false;return}const px=a.X(i),py=a.Y(r.scalars[k]);pen?a.x.lineTo(px,py):a.x.moveTo(px,py);pen=true});a.x.stroke()});document.querySelector('#scalarLegend').innerHTML=[...enabled].map(k=>`<span><i class="swatch" style="background:${colors[k]}"></i>${k}</span>`).join('')}
function drawVector(){const key=document.querySelector('#vector').value,st=selectedStages(),vals=DATA.filter(r=>st.has(r.stage)).flatMap(r=>r.vectors[key]);if(!vals.length){const {x,w,h}=size(document.querySelector('#vectorPlot'));x.fillStyle='#9eb0ca';x.fillText('Select at least one stage',w/2-70,h/2);return}const lo=Math.min(...vals),hi=Math.max(...vals),a=axes(document.querySelector('#vectorPlot'),lo,hi),n=Math.max(...DATA.map(r=>r.vectors[key].length));for(let b=0;b<n;b++){a.x.strokeStyle=`hsl(${(b*47)%360} 75% 62%)`;a.x.lineWidth=1;a.x.beginPath();let pen=false,lastStage=null;DATA.forEach((r,i)=>{if(!st.has(r.stage)||b>=r.vectors[key].length){pen=false;return}if(lastStage!==r.stage)pen=false;const px=a.X(i),py=a.Y(r.vectors[key][b]);pen?a.x.lineTo(px,py):a.x.moveTo(px,py);pen=true;lastStage=r.stage});a.x.stroke()}document.querySelector('#vectorTitle').textContent=key.replaceAll('_',' ')+' evolution'}
function drawHandoff(){document.querySelector('#handoff').innerHTML='<table><tr><th>Handoff</th><th>Vector</th><th>Result</th><th>Different buckets</th></tr>'+HANDOFF.flatMap(h=>Object.entries(h.vectors).map(([name,v])=>`<tr><td>${h.source_stage}m → ${h.target_stage}m</td><td>${name}</td><td style="color:${v.matches?'#69db9c':'#ff7373'}">${v.matches?'PASS':'FAIL'}</td><td>${v.different_buckets.join(', ')||'—'}</td></tr>`)).join('')+'</table>'}
function draw(){drawHandoff();drawScalars();drawVector()}document.querySelectorAll('input,select').forEach(x=>x.addEventListener('change',draw));addEventListener('resize',draw);draw();</script></body></html>'''


def render(records: list) -> str:
    return (HTML.replace("__DATA__", json.dumps(records, separators=(",", ":")))
            .replace("__HANDOFF__", json.dumps(build_handoff_audit(records),
                                                separators=(",", ":"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = build_records(json.loads(args.history.read_text()), args.config_root)
    if not records:
        raise SystemExit("No successful 6m, 9m, or 12m candidates found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(records))
    print(f"Wrote {len(records)} candidates to {args.output}")


if __name__ == "__main__":
    main()
