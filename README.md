# PhysLab — Neural Network Pruning and Information Processing Ability

> Phase transition in the Information Processing Ability (IPA) of neural networks under **random unstructured pruning**.

## Research Idea and Motivation

Neural networks store information in their weights. Random unstructured pruning zeros out a fraction of those weights at initialization — an artificial analogue of random synaptic loss in biological neurons — and we measure how the network's ability to learn degrades as that fraction grows.

The hypothesis driving this project: **there exists a critical pruning threshold beyond which IPA exhibits a sharp decline** (a phase transition), rather than a smooth monotonic falloff. By systematically sweeping pruning percentage (P%) across architectures and datasets, we aim to localize this transition and characterize how it depends on model capacity and data complexity. A secondary motivation is to ask whether IPA trends observed in artificial networks have correlates in biological neural systems, where synaptic loss is a known feature of aging and disease.

## IPA Methodology

Information Processing Ability (IPA) captures both **learning effectiveness** (how much the loss actually decreases) and **learning speed** (how many batches it took to get there).

- **Baseline CE**: `CE_0 = ln(10)` for 10-class classification (random guessing).
- **Asymptotic CE (`CE_asy`)**: currently computed as the mean of the last 20 cross-entropy values.
- **Cross-Entropy Learned**: `CE_L = CE_0 − CE_asy` — how much loss the network actually removed.
- **Batch Number Learned (`BN_L`)**: the batch index at which CE_Test reaches `CE_0 − 0.9·(CE_0 − CE_asy)`, i.e. the point where learning has effectively plateaued.
- **IPA**: `IPA = CE_L / BN_L`.

### Caveats (open questions being worked on)

- The "mean of the last 20 values" definition of `CE_asy` works cleanly only for **SLP**. A universal asymptote definition for CNN (and later DenseNet) is still an open problem — curve-fitting (`A + B/(x+1)^n`) is being evaluated as a generalization.
- The IPA formula above is expected to need a **batch-size-dependent factor**, which is still to be verified experimentally.

## Experimental Setup

| Component | Value |
| --- | --- |
| Architectures | SLP, CNN (DenseNet planned, out of scope here) |
| Datasets | SLP: MNIST, Fashion-MNIST · CNN: MNIST, Fashion-MNIST, CIFAR-10 |
| Pruning method | `torch.nn.utils.prune.random_unstructured`, applied at initialization and kept permanent for the run |
| Layer options (`PRUNE_LAYERS_OPTIONS`) | `'CONV'`, `'FHL'`, `'SHL'`, `'FHL+SHL'`, `'ALL'` (SLP uses `'ALL'` only) |
| Pruning percentages | 0 %, 10 %, …, 100 %, with fine-grained steps (82, 84, 86, 88, 92, 94, 96, 98) near the transition |
| Batch sizes | SLP: 64, 1024, 60000 (full-batch) · CNN: 64, 1024 |
| Runs per configuration | 100 |
| Optimizer / loss | Adadelta / Cross-Entropy |
| Stopping criterion | Relative change in mean `CE_Test` over consecutive 20-batch windows < 1 % |

Dataset-specific normalization (MNIST: μ=0.1307, σ=0.3081; Fashion-MNIST: μ=0.2860, σ=0.3530) is used in preference to fixed 0.5 normalization — it yields lower final CE and higher IPA.

## Repository Structure

```
physlab/
├── SLP/                                  # SLP training scripts + analysis notebooks
│   ├── SLP-MNIST/                        #   MNIST experiments
│   └── SLP-FMNIST/                       #   Fashion-MNIST experiments
├── Convolution/Convolutional/            # CNN training scripts + analysis notebooks
│   ├── Convolutional-MNIST/
│   ├── Convolutional-FMNIST/
│   └── Convolutional-CIFAR-10/
├── new_graph/Graphs/                     # All publication-ready figures (see layout below)
├── paper/                                # Draft paper: Neural research.docx
├── DenseNet/                             # Out of scope (not yet started)
└── new_test/                             # Out of scope (scratch/unused)
```

Each `SLP-*` and `Convolutional-*` directory contains:
- The training script (`.py`) that produces `.txt` raw logs into nested `prune_layers_*/p-percentage_*/batch_size_*/` directories.
- Task-specific Jupyter notebooks that read those `.txt` files and produce the figures under `new_graph/`.

## `new_graph/Graphs/` Layout

Figures are organized **model × dataset × pruning range**. The range subfolders exist because the phase transition lives in a narrow window (≈80 % for SLP, ≈90 % for CNN), so we render one "wide-view" plot (0–100 %) plus zoomed views that show the transition clearly.

### SLP — transition near 80 %

```
new_graph/Graphs/SLP/{SLP-MNIST, SLP-FMNIST}/
├── AUC_Average-CE/
│   ├── AUC_data_{0-100, 0-80-100, 80-100}/      # per-range AUC CSVs
│   └── AUC_graph_{0-100, 0-80-100, 80-100}/     # per-range AUC plots
├── Average_P%-BN/                               # averaged CE / accuracy vs batch, wide view
├── Avg_P%-BN-mapped_80-100/                     # 80–100 % zoom, remapped axis
├── Avg_P%-BN-only_80-100/                       # 80–100 % zoom, native axis
├── CE-Accuracy_P%-BN/
│   └── SLP_raw_plot_{64, 1024, 60000}/          # raw per-run curves at each batch size
└── (SLP-FMNIST only) IPA_CEL_BNL_fitting-function/   # per-run power-law fits (method under evaluation)
```

### CNN — transition near 90 %

```
new_graph/Graphs/Convolutional/{Convolution-MNIST, Convolutional-FMNIST, Convolutional_CIFAR-10}/
├── AUC_Average-CE/ (or AUC_Average_CE/)
│   ├── AUC_data_{0-100, 0-90-100, 90-100}/
│   └── AUC_graph_{0-100, 0-90-100, 90-100}/
├── Conv_Avg_Plots/
│   └── Conv_Avg_Plots_{0-100, 0-90-100, 90-100}/
└── Conv_Raw_Plots/
    └── Conv_raw_plot_{64, 1024}/                # CIFAR-10 has no raw-plots folder yet
```

## File Conventions

| Extension / Location | Role | Open? |
| --- | --- | --- |
| `.py` | Runs experiments, writes `.txt` raw logs | Yes |
| `.ipynb` | Generates figures; each notebook has a specific role | Yes |
| `.txt` | Raw CE / accuracy / batch-number logs | **No** (very large) |
| Image files (anywhere — `new_graph/`, and also inside `SLP/` and `Convolution/` subtrees) | Committed figures | **No** (reference by path only) |

Raw output columns (inside each `.txt`):
```
Current_Epoch | Batch/Total | CE_Train | Accuracy(%) | CE_Test | Batch_Number
```

## Running an Experiment

Scripts expect to be run from their own directory (output paths are relative):

```bash
cd SLP/SLP-MNIST && python SLP-MNIST-CE-STOP.py
cd SLP/SLP-FMNIST && python SLP-FMNIST-CE-STOP.py
cd Convolution/Convolutional/Convolutional-MNIST && python convolutional-MNIST-CE-stop.py
```

Key knobs at the top of `main()` in each script:

```python
PRUNE_LAYERS_OPTIONS = ['ALL']                        # SLP: only 'ALL'; CNN adds 'CONV','FHL','SHL','FHL+SHL'
ACCEPTABLE_PRUNE_PERCENTAGES = [i/100 for i in range(0, 110, 10)]
ACCEPTABLE_BATCH_SIZES = [64, 60000]
num_runs = 100
```

## Current Status and Ongoing Work

**Complete (stopping-criteria IPA pipeline)**
- SLP on MNIST and Fashion-MNIST: averaged CE / accuracy curves, AUC tables and plots, and raw per-run plots committed under `new_graph/Graphs/SLP/`.
- CNN on MNIST, Fashion-MNIST, and CIFAR-10: averaged CE plots and AUC tables/plots committed under `new_graph/Graphs/Convolutional/`.

**In progress (no results committed yet)**
- **Curve-fitting asymptote** — fitting each per-run CE curve to `A + B/(x+1)^n` and using `A` as `CE_asy`. Goal: a universal asymptote that works for CNN (and eventually DenseNet), not just SLP. Current work lives in `SLP-MNIST/fitting_function_IPA.ipynb` and `SLP-FMNIST/` fitting notebooks.
- **AUC-based IPA** — using area under the CE-vs-batch curve as an alternative single-scalar IPA measure.
- **Batch-size factor** — an explicit batch-size-dependent correction to the IPA formula still to be verified experimentally.

**Not yet committed**
- IPA-vs-P% plots themselves (the final publication figures) — planned for the next pass once the asymptote method stabilizes.

**Notes**
- SLP on CIFAR-10 was anomalous (learning only between P% = 90–94 %) and is not pursued further; SLP on CIFAR-100 did not learn at all and was dropped after a single sanity run.
- `DenseNet/` and `new_test/` are intentionally excluded from this revision: DenseNet work has not started, and `new_test/` is scratch space.

## Requirements

Python 3, PyTorch, torchvision. CUDA optional.
