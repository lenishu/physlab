"""Export annotated connectivity as CSR W[post, pre] plus its exact ID mapping."""
from pathlib import Path
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather
from scipy.sparse import coo_matrix, save_npz

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, default=BASE / 'data')
parser.add_argument('--output-dir', type=Path, default=BASE / 'matrices')
parser.add_argument('--superclass', help='Optional exact annotation superclass to select')
parser.add_argument('--status', default='Traced', help='Annotation status to retain; default Traced; use ALL to include every annotated segment')
args = parser.parse_args()
ann = feather.read_table(next(args.data_dir.glob('body-annotations*.feather'))).to_pandas()
if args.status != 'ALL': ann = ann[ann.status == args.status]
if args.superclass: ann = ann[ann.superclass == args.superclass]
ids = np.sort(ann.bodyId.unique().astype(np.int64))
if not len(ids): raise SystemExit('No annotated bodies match the selection.')
rows, columns, values = [], [], []
with pa.memory_map(str(next(args.data_dir.glob('connectome-weights*.feather'))), 'r') as source:
    reader = ipc.open_file(source)
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        pre = batch.column('body_pre').to_numpy(); post = batch.column('body_post').to_numpy()
        p, q = np.searchsorted(ids, pre), np.searchsorted(ids, post)
        valid = (p < len(ids)) & (q < len(ids))
        valid &= ids[np.minimum(p,len(ids)-1)] == pre
        valid &= ids[np.minimum(q,len(ids)-1)] == post
        rows.append(q[valid].astype(np.int32)); columns.append(p[valid].astype(np.int32))
        values.append(batch.column('weight').to_numpy()[valid].astype(np.float32))
matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                    shape=(len(ids),len(ids))).tocsr()
matrix.sum_duplicates()
args.output_dir.mkdir(parents=True, exist_ok=True)
save_npz(args.output_dir / 'synapse_counts_post_pre.npz',matrix)
np.save(args.output_dir / 'body_ids.npy',ids)
print(f'Shape={matrix.shape}, nnz={matrix.nnz:,}; W[post, pre], unsigned synapse counts')
