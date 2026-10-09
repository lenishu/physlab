"""Generate a sampled spatial overview and full traced-neuron group summaries."""
from pathlib import Path
import argparse
import json
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ann = feather.read_table(next(args.data_dir.glob('body-annotations*'))).to_pandas()
ann = ann[ann.status == 'Traced'].sort_values('bodyId').copy()
definitions = [
    ('Optic-lobe cells', 'ol_', 'Local visual circuitry and sensory cells'),
    ('Central-brain cells', 'cb_', 'Central-brain intrinsic, sensory and output classes'),
    ('Ventral nerve cord cells', 'vnc_', 'Nerve-cord intrinsic, sensory and motor classes'),
    ('Visual links', 'visual_', 'Visual projection and centrifugal cells'),
    ('Brain–cord links', '', 'Ascending, descending and related connector classes'),
    ('Other / unclassified', '', 'Other traced cells or missing superclass labels')]

def group(label):
    if not isinstance(label, str): return 5
    for i in range(4):
        if label.startswith(definitions[i][1]): return i
    if 'ascending' in label or 'descending' in label: return 4
    return 5

ann['family'] = ann.superclass.map(group)
nt = feather.read_table(next(args.data_dir.glob('body-neurotransmitters*')),
                        columns=['body', 'consensus_nt']).to_pandas().drop_duplicates('body')
ann = ann.merge(nt, left_on='bodyId', right_on='body', how='left')
groups = []
points = []
rng = np.random.default_rng(783)
for i, (name, prefix, role) in enumerate(definitions):
    part = ann[ann.family == i]
    somas = part[part.somaLocation.map(lambda a: a is not None and len(a) == 3)]
    n = min(len(somas), max(250, round(len(somas)/140024*16000)))
    if n:
        selected = somas.iloc[rng.choice(len(somas), n, replace=False)]
        points.extend([[round(float(x), 0), round(float(y), 0), round(float(z), 0), i]
                       for x,y,z in selected.somaLocation])
    groups.append({'name':name, 'role':role, 'count':len(part), 'located':len(somas),
                   'sampled':n, 'classes':part.superclass.fillna('Unclassified').value_counts().to_dict(),
                   'types':part.type.fillna('Unspecified').value_counts().head(8).to_dict(),
                   'transmitters':part.consensus_nt.fillna('Unknown').value_counts().to_dict()})

# Actual group-to-group synapse counts, from all connections whose endpoints are Traced.
ids = ann.bodyId.to_numpy(); family = ann.family.to_numpy()
order = np.argsort(ids); ids = ids[order]; family = family[order]
syn = np.zeros((6,6),dtype=np.int64); edges = np.zeros((6,6),dtype=np.int64)
with pa.memory_map(str(next(args.data_dir.glob('connectome-weights*'))),'r') as source:
    reader = ipc.open_file(source)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        pre = b.column('body_pre').to_numpy(); post = b.column('body_post').to_numpy()
        p,q = np.searchsorted(ids,pre),np.searchsorted(ids,post)
        valid = (p<len(ids)) & (q<len(ids))
        valid &= ids[np.minimum(p,len(ids)-1)] == pre
        valid &= ids[np.minimum(q,len(ids)-1)] == post
        code = family[p[valid]]*6+family[q[valid]]
        edges += np.bincount(code, minlength=36).reshape(6,6)
        np.add.at(syn.ravel(),code,b.column('weight').to_numpy()[valid])
        if i%400==0: print(f'Connectivity batch {i+1}/{reader.num_record_batches}',flush=True)
data = {'groups':groups, 'points':points, 'synapses':syn.tolist(), 'edges':edges.tolist(),
        'traced':len(ann), 'located':sum(g['located'] for g in groups), 'sampled':len(points),
        'source':'MaleCNS v1.0; status Traced; superclass-based circuit families; somaLocation coordinates'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
print(json.dumps({k:v for k,v in data.items() if k not in ['points','groups']},indent=2))
