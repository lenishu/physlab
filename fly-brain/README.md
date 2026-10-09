# Fly brain: inspect first, then test MNIST learning and pruning

This folder is an exploratory addition to PhysLab. The existing SLP/CNN experiments are separate.

## Open the downloaded files

Feather is an Apache Arrow binary table format. A browser cannot display it as a spreadsheet. Use the HTML report for a first look, or Python to read selected columns. The files are in `data/`.

1. Open `reports/index.html` in any browser. It shows measured sizes, row counts, schemas, first-row previews, connection histograms, neuron classes, neurotransmitter labels, and a small directed graph and heatmap. The node table can be searched. Plots are selected samples, not a 3D brain reconstruction.
2. Open exported `*-preview.csv` tables in Excel, VS Code, or a text editor. These contain only the first 1,000 rows, not the whole dataset.
3. For anatomical 3D exploration, use the official [MaleCNS explorer](https://male-cns.janelia.org/) and its Neuroglancer / neuPrint links.

### Set up Python (PowerShell, from this directory)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe visualize.py
Start-Process .\reports\index.html
```

If `python` selects the wrong installation, use your Miniconda Python explicitly. No Streamlit server is required: the report is a standalone local HTML file with adjacent PNGs.

### Inspect a table directly

```python
import pyarrow.feather as feather

annotations = feather.read_table(
    'data/body-annotations-male-cns-v1.0-minconf-0.5.feather'
).to_pandas()
print(annotations[['bodyId', 'type', 'superclass', 'somaSide']].head())

# This reads the selected columns for the ENTIRE connectivity table.
# Prefer the batch-based viewer for low-memory inspection.
weights = feather.read_table(
    'data/connectome-weights-male-cns-v1.0-minconf-0.5.feather',
    columns=['body_pre', 'body_post', 'weight'], memory_map=True
)
print(weights.slice(0, 10).to_pandas())
```

To inspect all incident connections of one body ID (use an ID from the report):

```powershell
.\.venv\Scripts\python.exe visualize.py --body 12345
```

The output is `reports/body-12345-connections.csv`, including incoming and outgoing rows.

## What the data represents

- Connectivity: `body_pre -> body_post`, with `weight` equal to the number of anatomical synapses. Multiple synapses between a pair form one weighted connection. This is not a pretrained MNIST checkpoint.
- Annotations: `bodyId`, cell types, superclasses, sides, and related labels. These 211,577 annotated bodies include 165,122 with status `Traced`, plus glia, orphan fragments and other statuses. The viewer's annotated-endpoint statistics include all annotation statuses. For a neuron model the export defaults to status `Traced`; inspect excluded sensory/anchor categories before selecting a biological circuit. The full connectivity table also includes segments outside this annotation set.
- Neurotransmitters: the join key is `body`, matching annotation `bodyId`. `predicted_nt`, confidence, and `consensus_nt` are predictions/labels, not electrical weights. Do not automatically treat all positive counts as excitation or equate a transmitter with a universal sign; functional effects depend on receptors and modeling assumptions.
- IDs are not contiguous array indices. Save an ID-to-index mapping with every matrix.

## Export the weight matrix

```powershell
.\.venv\Scripts\python.exe export_sparse.py
```

This keeps edges with both endpoints in annotations whose status is `Traced` and saves:

- `matrices/synapse_counts_post_pre.npz`: SciPy sparse CSR, unsigned float32 counts, orientation `W[post, pre]`.
- `matrices/body_ids.npy`: index `i` corresponds to body ID `body_ids[i]`.

Use `--superclass EXACT_LABEL` to export the induced graph of one superclass after checking available labels in the viewer. That selection excludes cross-class edges; it is not a full visual pathway. Use `--status ALL` only if you deliberately want every annotated segment, including glia/fragments. Full export may need several GB of RAM because it collects sparse arrays and converts COO to CSR. Never call `toarray()` on the full-brain matrix: a 166k-square float32 array alone is about 110 GB.

## Plan A: a small, interpretable MNIST pilot

1. **Define a reproducible subgraph.** Start with 1,000-5,000 annotated neurons from a documented circuit or a deterministic selection. Save IDs, edges, dataset version and selection method. A useful visual circuit can span several superclasses; do not assume the strongest-edge plot is a representative brain model.
2. **Build a sparse recurrent rate model.** Flatten the 28x28 image, then use a fixed random input projection into selected neurons. For T recurrent steps, compute `h_next = (1-alpha)*h + alpha*tanh(g*W@h + B@x)`. Reset hidden state for each image. Normalize W and tune gain/steps on a validation set; count magnitudes alone do not define stable dynamics. Begin explicitly with an unsigned structural model, then test signed variants as a separate assumption.
3. **Train only a ten-class readout.** Train `logits = C@h + bias` with cross-entropy. Keep W and B fixed. This tests feature usefulness, not biological learning. Compare against a random reservoir with matched size, directed edge count, weight distribution and dynamics; also include degree-preserving directed rewiring to isolate topology.
4. **Use fixed train/validation/test splits.** Reserve validation data from MNIST's training split. Tune without using test results. Use identical image preprocessing, input projections, readout sizes, optimization budget and multiple matched seeds across conditions. Report test cross-entropy, accuracy, training time and memory.
5. **Sanity checks.** A trainable linear classifier on pixels must learn; shuffled labels should fail to generalize; outputs must remain finite; ensure no recurrent state leaks between examples. Inspect how much information the input adapter alone preserves.

## Plan B: the PhysLab pruning / IPA experiment

Run two distinct protocols; they answer different questions.

**Pruning before training:** remove a sampled fraction of recurrent connections at initialization, keep the mask permanent, and retrain the readout from scratch. Use a coarse 0, 20, 40, 60, 80, 90, 95, 99, 100% sweep, then refine near any observed change. Keep input and readout adapters intact for this initial experiment. Begin with 5-10 paired seeds for feasibility; increase to the project's 100-run convention only after the pilot works.

**Damage after training:** train the intact model, then apply increasingly severe connection removal without retraining. Measure immediate loss/accuracy changes; optionally measure recovery with readout retraining separately.

Pruning an edge removes a neuron-to-neuron connection and all its counted synapses. This differs from removing individual anatomical synapses: for that experiment use `surviving_count ~ Binomial(original_count, 1-p)` and retain edges with surviving count > 0. Also test neuron removal separately. State the lesion unit in every plot.

Use paired/nested masks across pruning levels for each seed. Compare fly topology to matched randomized controls under the same protocol. Fix the normalization policy in advance: leaving intact-model scaling fixed exposes activity collapse from damage; renormalizing each pruned model partially compensates for damage and must be labeled as a separate condition.

Log batch number, CE_Train, CE_Validation, accuracy and wall time; evaluate CE_Test at defined checkpoints without using it for tuning or early stopping. For compatibility with PhysLab, analyze `CE_L = ln(10) - CE_asy` and `IPA = CE_L / BN_L` with a stated asymptote estimator. Also record measured initial CE: it need not equal ln(10). BN_L is undefined if there is no positive, reliably estimated loss reduction; report those cases explicitly instead of dividing by zero. Add uncertainty intervals and retain raw learning curves. A sharp transition is a hypothesis, not guaranteed by the connectome.

## Plan C: trainable fly topology

After the fixed-reservoir pilot, train weights only on the existing edge mask, with the same architecture and masks initialized either from normalized synapse counts or random values. This tests the topology and initialization rather than claiming to reproduce a fly's learning rule. Compare to an equally sized randomized sparse network; a conventional MLP/CNN is a useful MNIST reference but not a resource-matched topology control.

## Provenance

Dataset: MaleCNS v1.0, FlyEM / HHMI Janelia Research Campus, University of Cambridge, MRC Laboratory of Molecular Biology, Google Research. [Official downloads and license](https://male-cns.janelia.org/download/). Released under CC BY; preserve attribution when distributing derivatives. Files were downloaded by the user from the official public Google Storage bucket. `data-manifest.json` records local file sizes and SHA-256 hashes. No MNIST training has been run by this viewer.
