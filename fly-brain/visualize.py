"""Inspect MaleCNS Feather files and generate a local HTML report.

Reads connectivity in Arrow record batches; never builds a dense full-brain matrix.
The plotted graph is a deliberately selected subset, not an anatomical rendering.
"""
from pathlib import Path
import argparse
import heapq
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather

BASE = Path(__file__).resolve().parent


def inspect(path):
    with pa.memory_map(str(path), 'r') as source:
        reader = ipc.open_file(source)
        rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        first = reader.get_batch(0).slice(0, 8).to_pandas()
        return {'file': path.name, 'bytes': path.stat().st_size, 'rows': rows,
                'columns': [{'name': f.name, 'type': str(f.type)} for f in reader.schema],
                'preview': first.astype(str).to_dict('records')}


def bar(series, title, path):
    series = series.fillna('Unknown').replace('', 'Unknown').value_counts().head(18)
    fig, ax = plt.subplots(figsize=(10, 6))
    series.sort_values().plot.barh(ax=ax, color='#227c9d')
    ax.set(title=title, xlabel='Rows'); fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=BASE / 'data')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'reports')
    parser.add_argument('--body', type=int, help='Also export all incoming/outgoing rows for this body ID')
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data_dir.glob('*.feather'))
    if len(files) != 3:
        raise SystemExit('Expected exactly the three MaleCNS .feather files in data/.')
    metadata = [inspect(p) for p in files]
    annotation_path = next(p for p in files if p.name.startswith('body-annotations'))
    nt_path = next(p for p in files if p.name.startswith('body-neurotransmitters'))
    weights_path = next(p for p in files if p.name.startswith('connectome-weights'))
    annotations = feather.read_table(annotation_path).to_pandas()
    nt = feather.read_table(nt_path).to_pandas()
    annotations.to_csv(out / 'neuron_annotations.csv', index=False)
    nt.head(1000).to_csv(out / 'neurotransmitter_preview.csv', index=False)
    for p in files:
        with pa.memory_map(str(p), 'r') as source:
            ipc.open_file(source).get_batch(0).slice(0, 1000).to_pandas().to_csv(
                out / (p.stem + '-preview.csv'), index=False)

    # Curated annotations identify candidate neurons. An ID is NOT a matrix index.
    ids = np.sort(annotations.bodyId.dropna().unique().astype(np.int64))
    incoming = np.zeros(len(ids), dtype=np.int64)
    outgoing = np.zeros(len(ids), dtype=np.int64)
    in_strength = np.zeros(len(ids), dtype=np.int64)
    out_strength = np.zeros(len(ids), dtype=np.int64)
    bins = np.geomspace(1, 1e7, 71)
    histogram = np.zeros(len(bins)-1, dtype=np.int64)
    top_edges = []
    rows = synapses = annotated_rows = annotated_synapses = self_rows = zero_rows = 0
    minimum = None; maximum = 0
    body_parts = []
    with pa.memory_map(str(weights_path), 'r') as source:
        reader = ipc.open_file(source)
        for b in range(reader.num_record_batches):
            batch = reader.get_batch(b)
            pre = batch.column('body_pre').to_numpy()
            post = batch.column('body_post').to_numpy()
            weight = batch.column('weight').to_numpy()
            rows += len(weight); synapses += int(weight.sum())
            self_rows += int((pre == post).sum()); zero_rows += int((weight <= 0).sum())
            minimum = int(weight.min()) if minimum is None else min(minimum, int(weight.min()))
            maximum = max(maximum, int(weight.max()))
            histogram += np.histogram(weight, bins=bins)[0]
            pi, qi = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            valid = (pi < len(ids)) & (qi < len(ids))
            valid &= (ids[np.minimum(pi, len(ids)-1)] == pre)
            valid &= (ids[np.minimum(qi, len(ids)-1)] == post)
            p, q, w = pi[valid], qi[valid], weight[valid]
            annotated_rows += len(w); annotated_synapses += int(w.sum())
            np.add.at(outgoing, p, 1); np.add.at(incoming, q, 1)
            np.add.at(out_strength, p, w); np.add.at(in_strength, q, w)
            # Only retain the strongest 3,000 annotated edges, bounded memory.
            candidates = np.flatnonzero(valid & (pre != post))
            if len(candidates) > 3000:
                candidates = candidates[np.argpartition(weight[candidates], -3000)[-3000:]]
            for k in candidates:
                edge = (int(weight[k]), int(pre[k]), int(post[k]))
                if len(top_edges) < 3000: heapq.heappush(top_edges, edge)
                elif edge > top_edges[0]: heapq.heapreplace(top_edges, edge)
            if args.body is not None:
                match = (pre == args.body) | (post == args.body)
                if match.any(): body_parts.append(batch.filter(pa.array(match)).to_pandas())
            if b % 100 == 0: print(f'Batch {b+1}/{reader.num_record_batches}; {rows:,} connections', flush=True)
    stats = {'all_segment_connection_rows': rows, 'all_segment_synapse_count_sum': synapses,
             'annotation_rows': len(annotations), 'unique_annotated_body_ids': len(ids),
             'connections_with_both_endpoints_annotated': annotated_rows,
             'synapse_sum_between_annotated_endpoints': annotated_synapses,
             'minimum_weight': minimum, 'maximum_weight': maximum,
             'self_connection_rows': self_rows, 'nonpositive_weight_rows': zero_rows}
    pd.DataFrame({'bodyId': ids, 'in_degree': incoming, 'out_degree': outgoing,
                  'incoming_synapse_sum': in_strength, 'outgoing_synapse_sum': out_strength}).to_csv(
                      out / 'annotated_connectivity_metrics.csv', index=False)
    if args.body is not None:
        body_df = pd.concat(body_parts, ignore_index=True) if body_parts else pd.DataFrame(columns=['body_pre','body_post','weight'])
        body_df.sort_values('weight', ascending=False).to_csv(out / f'body-{args.body}-connections.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stairs(histogram, bins, fill=True, color='#227c9d')
    ax.set(xscale='log', yscale='log', xlabel='Synapses per directed segment pair',
           ylabel='Connection rows per logarithmic bin', title='Connection strength distribution: all segments')
    fig.tight_layout(); fig.savefig(out / 'weights.png', dpi=140); plt.close(fig)
    bar(annotations['superclass'], 'Annotated neuron superclasses', out / 'classes.png')
    bar(nt['consensus_nt'], 'Consensus neurotransmitter labels (all table rows)', out / 'neurotransmitters.png')

    ordered = sorted(top_edges, reverse=True)
    nodes = []
    for w, pre, post in ordered:
        for node in (pre, post):
            if node not in nodes and len(nodes) < 60: nodes.append(node)
        if len(nodes) == 60: break
    lookup = {n:i for i,n in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    edges = [(w,p,q) for w,p,q in ordered if p in lookup and q in lookup]
    for w,p,q in edges: matrix[lookup[q], lookup[p]] += w
    # This is a strongest-edge sample; other edges between these nodes may be omitted.
    fig, ax = plt.subplots(figsize=(9,8))
    im = ax.imshow(np.log1p(matrix), cmap='magma')
    ax.set(title='Strongest-edge sample: 60 annotated bodies', xlabel='Presynaptic node index', ylabel='Postsynaptic node index')
    fig.colorbar(im, ax=ax, label='log(1 + synapse count)'); fig.tight_layout()
    fig.savefig(out / 'matrix.png', dpi=140); plt.close(fig)
    theta = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    fig, ax = plt.subplots(figsize=(10,10))
    for w,p,q in edges[:150]:
        ax.add_patch(FancyArrowPatch(xy[lookup[p]], xy[lookup[q]], arrowstyle='-|>',
            mutation_scale=7, alpha=.22, linewidth=.5 + 2*w/max(e[0] for e in edges), color='#227c9d'))
    ax.scatter(xy[:,0],xy[:,1],s=60,color='#e87524',zorder=3)
    for i,(x,y) in enumerate(xy): ax.text(x*1.1,y*1.1,str(i),ha='center',va='center',fontsize=8)
    ax.set(title='Top 150 sampled edges; circular layout, not anatomy', aspect='equal', xlim=(-1.3,1.3), ylim=(-1.3,1.3)); ax.axis('off')
    fig.tight_layout(); fig.savefig(out / 'graph.png', dpi=140); plt.close(fig)
    node_table = pd.DataFrame({'plot_index':range(len(nodes)), 'bodyId':nodes}).merge(annotations, on='bodyId', how='left')
    node_table.to_csv(out / 'plotted_nodes.csv', index=False)
    (out / 'summary.json').write_text(json.dumps({'statistics':stats,'files':metadata}, indent=2), encoding='utf-8')

    sections = []
    for item in metadata:
        schema = pd.DataFrame(item['columns']).to_html(index=False, escape=True)
        preview = pd.DataFrame(item['preview']).to_html(index=False, escape=True)
        sections.append(f'<h3>{html.escape(item["file"])}</h3><p>{item["rows"]:,} rows; {item["bytes"]:,} bytes</p>{schema}<details><summary>Show first eight rows</summary><div class="scroll">{preview}</div></details>')
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>Fly brain data explorer</title>
    <style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;background:#f7fafc;color:#173047}h1,h2{color:#125b75}section{background:white;padding:24px;border-radius:14px;margin:20px 0}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border:1px solid #dae4ec;text-align:left}img{max-width:100%}.scroll{overflow:auto}summary{cursor:pointer}input{padding:10px;width:350px}</style></head><body>
    <h1>Fly brain data explorer</h1><p>MaleCNS v1.0 · structural connectivity, annotations and neurotransmitter predictions</p>
    <section><h2>What these files mean</h2><p>Each connectivity row is a directed pair: body_pre → body_post. Weight is a synapse count, not a learned PyTorch parameter or measured electrical conductance. Body IDs are labels, not consecutive array positions. The full graph includes segments beyond the annotation table. Annotated bodies include glia, orphan fragments and other statuses; these viewer statistics include all annotation statuses. For neuron models export_sparse.py defaults to status Traced.</p><p>The matrix convention for a future model is W[post, pre], so activity propagates as W @ activity. Neurotransmitter predictions need biological modeling assumptions before converting counts to signed functional weights.</p></section>
    <section><h2>Measured file statistics</h2>STATS</section>
    <section><h2>Files, columns and previews</h2>SCHEMA</section>
    <section><h2>Connection strengths</h2><img src="weights.png"></section>
    <section><h2>Neuron classes and neurotransmitter labels</h2><img src="classes.png"><img src="neurotransmitters.png"></section>
    <section><h2>A small view of connectivity</h2><p>We retain the strongest 3,000 annotated connections, then choose up to 60 bodies from the strongest pairs. Both plots use only this retained edge sample; they are not the complete induced subgraph. Node selection is biased toward strong connections. Circular positions do not represent locations in the brain.</p><img src="graph.png"><img src="matrix.png"><h3>Find a plotted body or type</h3><input id="search" placeholder="Search body ID, class or neuron type"><div class="scroll" id="nodes">NODES</div></section>
    <section><h2>Exported tables</h2><p><a href="annotated_connectivity_metrics.csv">Annotated degree and strength metrics</a> · <a href="neuron_annotations.csv">Neuron annotations</a> · <a href="plotted_nodes.csv">Plotted node mapping</a> · <a href="summary.json">Summary and schemas</a></p><p>For any body: python visualize.py --body BODY_ID. This exports all incident connectivity rows to CSV.</p><p>Source: <a href="https://male-cns.janelia.org/download/">FlyEM / HHMI Janelia, Cambridge, MRC LMB and Google Research</a>. Dataset licensed CC BY.</p></section>
    <script>document.querySelector('#search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('#nodes tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});</script></body></html>'''
    page = page.replace('STATS', pd.DataFrame(list(stats.items()), columns=['Measurement','Value']).to_html(index=False))
    page = page.replace('SCHEMA', ''.join(sections))
    page = page.replace('NODES', node_table[['plot_index','bodyId','type','superclass','somaSide']].to_html(index=False,escape=True))
    (out / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps(stats,indent=2)); print(f'Report: {out / "index.html"}')


if __name__ == '__main__': main()

