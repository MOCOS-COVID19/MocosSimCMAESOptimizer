#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from pathlib import Path


def categorize_plot(stem: str) -> str:
    if stem.startswith('infection_modulation_'):
        return 'infection_modulation'
    if stem.startswith('mild_detection_modulation_'):
        return 'mild_detection_modulation'
    if stem.startswith('tracing_modulation_'):
        return 'tracing_modulation'
    if stem.startswith('imported_cases'):
        return 'imported_cases'
    return stem.split('_')[0]


def build_items(plots_dir: Path):
    real_sims_root = plots_dir.parent
    items = []

    for path in sorted(plots_dir.glob('*.png')):
        stem = path.stem
        match = re.search(r'\[(\d+)\]', stem)
        items.append(
            {
                'file': path.name,
                'label': stem.replace('_', ' '),
                'category': categorize_plot(stem),
                'order': int(match.group(1)) if match else -1,
                'metrics': None,
                'kind': 'plot',
                'stage_order': math.inf,
                'iter_order': math.inf,
                'cand_order': math.inf,
                'score_order': math.inf,
            }
        )

    for path in sorted(real_sims_root.glob('stage_*/iter_*/cand_*/gt_vs_sim*.png')):
        rel = os.path.relpath(path, plots_dir)
        metrics_path = path.with_name('metrics.json')
        metrics = None
        score = math.inf
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                raw_score = metrics.get('score')
                if isinstance(raw_score, (int, float)) and math.isfinite(raw_score):
                    score = float(raw_score)
            except Exception:
                metrics = {'error': 'failed to parse metrics.json'}

        stage, iteration, cand = path.parts[-4:-1]
        stage_match = re.search(r'(\d+)', stage)
        iter_match = re.search(r'(\d+)', iteration)
        cand_match = re.search(r'(\d+)', cand)
        variant = 'cumulative' if path.stem.endswith('_cumulative') else 'daily'
        items.append(
            {
                'file': rel,
                'label': f'{stage} / {iteration} / {cand} / {path.stem}',
                'category': 'gt_vs_sim',
                'order': 0,
                'metrics': metrics,
                'kind': 'gt_vs_sim',
                'stage_order': int(stage_match.group(1)) if stage_match else math.inf,
                'iter_order': int(iter_match.group(1)) if iter_match else math.inf,
                'cand_order': int(cand_match.group(1)) if cand_match else math.inf,
                'score_order': score,
                'variant': variant,
            }
        )

        if path.stem == 'gt_vs_sim':
            for metric_name in [
                'daily_detections',
                'daily_hospitalizations',
                'daily_deaths',
                'daily_student_detections',
                'daily_detections_cumulative',
                'daily_hospitalizations_cumulative',
                'daily_deaths_cumulative',
                'daily_student_detections_cumulative',
                'sax_scholars_rmae',
            ]:
                items.append(
                    {
                        'file': rel,
                        'label': f'{stage} / {iteration} / {cand} / {path.stem} · {metric_name}',
                        'category': 'gt_vs_sim_metric',
                        'order': 0,
                        'metrics': metrics,
                        'kind': 'gt_vs_sim_metric',
                        'stage_order': int(stage_match.group(1)) if stage_match else math.inf,
                        'iter_order': int(iter_match.group(1)) if iter_match else math.inf,
                        'cand_order': int(cand_match.group(1)) if cand_match else math.inf,
                        'score_order': score,
                        'variant': metric_name,
                    }
                )

    for path in sorted(real_sims_root.glob('stage_*/iter_*/cand_*/temporal_params_three_panels_local.png')):
        rel = os.path.relpath(path, plots_dir)
        metrics_path = path.with_name('metrics.json')
        metrics = None
        score = math.inf
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                raw_score = metrics.get('score')
                if isinstance(raw_score, (int, float)) and math.isfinite(raw_score):
                    score = float(raw_score)
            except Exception:
                metrics = {'error': 'failed to parse metrics.json'}

        stage, iteration, cand = path.parts[-4:-1]
        stage_match = re.search(r'(\d+)', stage)
        iter_match = re.search(r'(\d+)', iteration)
        cand_match = re.search(r'(\d+)', cand)
        items.append(
            {
                'file': rel,
                'label': f'{stage} / {iteration} / {cand} / {path.stem}',
                'category': 'temporal_params_local',
                'order': 0,
                'metrics': metrics,
                'kind': 'temporal_params_local',
                'stage_order': int(stage_match.group(1)) if stage_match else math.inf,
                'iter_order': int(iter_match.group(1)) if iter_match else math.inf,
                'cand_order': int(cand_match.group(1)) if cand_match else math.inf,
                'score_order': score,
            }
        )

    return items


HTML_TEMPLATE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a30; --muted:#93a4c3; --text:#e8eefc; --accent:#6ea8fe; --border:#25304d; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: Inter, system-ui, sans-serif; background:var(--bg); color:var(--text); overflow:hidden; }}
.layout {{ display:grid; grid-template-columns: 340px 1fr; height:100vh; }}
aside {{ border-right:1px solid var(--border); background:var(--panel); padding:16px; overflow:auto; }}
main {{ padding:16px; overflow:hidden; }}
input, select {{ width:100%; padding:10px 12px; margin:0 0 10px; border:1px solid var(--border); border-radius:10px; background:#0d1427; color:var(--text); }}
.grid {{ display:grid; gap:8px; max-height:calc(100vh - 180px); overflow:auto; padding-right:4px; }}
.card {{ border:1px solid var(--border); border-radius:12px; padding:10px 12px; cursor:pointer; background:#0d1427; }}
.card:hover, .card.active {{ border-color:var(--accent); background:#111b33; }}
.card .meta {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.viewer {{ display:grid; grid-template-rows:auto minmax(220px, 46vh) minmax(220px, 1fr); gap:12px; height:calc(100vh - 32px); min-height:0; }}
.hero {{ border:1px solid var(--border); border-radius:16px; background:var(--panel); padding:12px; min-height:0; }}
.hero img {{ width:100%; height:100%; object-fit:contain; display:block; border-radius:10px; background:white; }}
.toolbar {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:space-between; }}
.btns {{ display:flex; gap:8px; }}
button, a.btn {{ background:#17305c; color:var(--text); border:1px solid #28457c; padding:9px 12px; border-radius:10px; text-decoration:none; cursor:pointer; }}
small {{ color:var(--muted); }}
.bottom {{ display:grid; grid-template-columns:minmax(320px, 1.4fr) minmax(260px, 0.9fr); gap:12px; min-height:0; }}
.thumbgrid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(180px,1fr)); grid-auto-rows:190px; gap:10px; overflow:auto; min-height:180px; padding-right:4px; align-content:start; }}
.thumb {{ border:1px solid var(--border); background:var(--panel); border-radius:12px; overflow:hidden; cursor:pointer; display:grid; grid-template-rows:120px 1fr; min-height:190px; }}
.thumb img {{ width:100%; height:120px; object-fit:cover; display:block; background:white; }}
.thumb div {{ padding:8px; font-size:12px; }}
.metrics {{ border:1px solid var(--border); background:var(--panel); border-radius:12px; padding:12px; overflow:auto; min-height:0; }}
.metrics h3 {{ margin:0 0 10px; font-size:14px; }}
.metrics pre {{ margin:0; white-space:pre-wrap; word-break:break-word; font-size:12px; color:var(--muted); }}
@media (max-width: 1100px) {{ .bottom {{ grid-template-columns:1fr; }} }}
@media (max-width: 900px) {{ body {{ overflow:auto; }} .layout {{ grid-template-columns:1fr; height:auto; }} aside {{ border-right:none; border-bottom:1px solid var(--border); max-height:42vh; }} main {{ overflow:visible; }} .viewer {{ height:auto; grid-template-rows:auto auto auto; }} .bottom {{ grid-template-columns:1fr; }} .thumbgrid, .metrics {{ overflow:visible; }} }}
</style>
</head>
<body>
<div class="layout">
  <aside>
    <h2 style="margin-top:0">{title}</h2>
    <input id="search" placeholder="Search plots..." />
    <select id="category"><option value="">All categories</option></select>
    <small id="count"></small>
    <div class="grid" id="list"></div>
  </aside>
  <main>
    <div class="viewer">
      <div class="toolbar">
        <div>
          <h2 id="title" style="margin:0 0 4px"></h2>
          <small id="subtitle"></small>
        </div>
        <div class="btns">
          <button id="prev">Prev</button>
          <button id="next">Next</button>
          <a id="open" class="btn" target="_blank">Open image</a>
        </div>
      </div>
      <div class="hero"><img id="image" alt="plot preview" /></div>
      <div class="bottom">
        <div class="thumbgrid" id="thumbs"></div>
        <div class="metrics"><h3>Metrics</h3><pre id="metrics">Select a plot to view metrics.</pre></div>
      </div>
    </div>
  </main>
</div>
<script>
const items = {items_json};
const listEl = document.getElementById('list');
const thumbsEl = document.getElementById('thumbs');
const searchEl = document.getElementById('search');
const categoryEl = document.getElementById('category');
const countEl = document.getElementById('count');
const titleEl = document.getElementById('title');
const subtitleEl = document.getElementById('subtitle');
const imageEl = document.getElementById('image');
const openEl = document.getElementById('open');
const metricsEl = document.getElementById('metrics');
let filtered = items.slice();
let current = 0;
const categories = [...new Set(items.map(x => x.category))].sort();
for (const cat of categories) {{ const o = document.createElement('option'); o.value = cat; o.textContent = cat; categoryEl.appendChild(o); }}
function compareItems(a, b) {{
  const scoreCategories = new Set(['gt_vs_sim', 'temporal_params_local']);
  if (scoreCategories.has(a.category) && scoreCategories.has(b.category)) {{
    return (a.score_order - b.score_order) || (a.stage_order - b.stage_order) || (a.iter_order - b.iter_order) || (a.cand_order - b.cand_order) || (a.variant || '').localeCompare(b.variant || '') || a.label.localeCompare(b.label);
  }}
  if (a.category === 'gt_vs_sim' && b.category === 'gt_vs_sim') {{
    return (a.score_order - b.score_order) || (a.stage_order - b.stage_order) || (a.iter_order - b.iter_order) || (a.cand_order - b.cand_order) || a.variant.localeCompare(b.variant) || a.label.localeCompare(b.label);
  }}
  return a.category.localeCompare(b.category) || a.label.localeCompare(b.label);
}}
function renderList() {{
  const q = searchEl.value.toLowerCase().trim();
  const cat = categoryEl.value;
  filtered = items.filter(x => (!cat || x.category === cat) && (!q || x.label.toLowerCase().includes(q) || x.file.toLowerCase().includes(q)));
  filtered.sort(compareItems);
  if (current >= filtered.length) current = 0;
  listEl.innerHTML = '';
  thumbsEl.innerHTML = '';
  countEl.textContent = `${{filtered.length}} / ${{items.length}} items`;
  filtered.forEach((item, idx) => {{
    const scoreText = (item.category === 'gt_vs_sim' || item.category === 'temporal_params_local') && Number.isFinite(item.score_order) ? ` · score=${{item.score_order.toFixed(4)}}` : '';
    const card = document.createElement('div');
    card.className = 'card' + (idx === current ? ' active' : '');
    card.innerHTML = `<div>${{item.label}}</div><div class="meta">${{item.category}}${{scoreText}}</div>`;
    card.onclick = () => {{ current = idx; renderViewer(); renderList(); }};
    listEl.appendChild(card);
    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    thumb.innerHTML = `<img loading="lazy" src="${{encodeURI(item.file)}}" alt="${{item.label}}"><div>${{item.label}}${{scoreText}}</div>`;
    thumb.onclick = () => {{ current = idx; renderViewer(); renderList(); }};
    thumbsEl.appendChild(thumb);
  }});
  renderViewer();
}}
function renderViewer() {{
  if (!filtered.length) {{
    titleEl.textContent = 'No plots match'; subtitleEl.textContent = ''; imageEl.removeAttribute('src'); openEl.removeAttribute('href'); metricsEl.textContent = 'No metrics.'; return;
  }}
  const item = filtered[current];
  const scoreText = (item.category === 'gt_vs_sim' || item.category === 'temporal_params_local') && Number.isFinite(item.score_order) ? ` · score=${{item.score_order}}` : '';
  titleEl.textContent = item.label;
  subtitleEl.textContent = `${{item.category}}${{scoreText}} · ${{current+1}} / ${{filtered.length}}`;
  imageEl.src = encodeURI(item.file);
  openEl.href = encodeURI(item.file);
  metricsEl.textContent = item.metrics ? JSON.stringify(item.metrics, null, 2) : 'No metrics for this plot.';
}}
document.getElementById('prev').onclick = () => {{ if (!filtered.length) return; current = (current - 1 + filtered.length) % filtered.length; renderViewer(); renderList(); }};
document.getElementById('next').onclick = () => {{ if (!filtered.length) return; current = (current + 1) % filtered.length; renderViewer(); renderList(); }};
searchEl.oninput = renderList;
categoryEl.onchange = renderList;
window.addEventListener('keydown', (e) => {{ if (e.key === 'ArrowLeft') document.getElementById('prev').click(); if (e.key === 'ArrowRight') document.getElementById('next').click(); }});
renderList();
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots-dir', required=True, type=Path)
    parser.add_argument('--title', default='Plot explorer')
    args = parser.parse_args()

    plots_dir = args.plots_dir.resolve()
    items = build_items(plots_dir)
    html = HTML_TEMPLATE.format(title=args.title, items_json=json.dumps(items))
    out = plots_dir / 'index.html'
    out.write_text(html, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
