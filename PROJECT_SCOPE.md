# Project Scope — PhysLab: Neural Network Pruning & Information Processing Ability

## 1. Research Question

Does random unstructured pruning of neural network weights produce a **sharp phase transition** in a network's ability to learn, rather than a smooth gradual decline? And if so, where does that threshold lie, and how does it depend on architecture and dataset complexity?

The secondary question is whether such transitions in artificial networks are analogous to synaptic loss observed in biological neural systems (aging, disease).

---

## 2. Core Concept: Information Processing Ability (IPA)

IPA is the project's central metric — a single scalar that captures both *how much* a network learned and *how fast* it learned.

| Symbol | Definition |
|--------|-----------|
| `CE_0` | Baseline cross-entropy = `ln(10)` (random guessing on 10 classes) |
| `CE_asy` | Asymptotic CE — the network's final converged loss |
| `CE_L` | `CE_0 − CE_asy` — total loss reduction ("how much learned") |
| `BN_L` | Batch index where CE_Test reaches 90% of its total drop ("how fast learned") |
| **IPA** | `CE_L / BN_L` |

**Open problem:** `CE_asy` is cleanly defined (mean of last 20 values) for SLP, but a universal definition for CNN is still being worked out. Curve-fitting (`A + B/(x+1)^n`) is the leading candidate.

---

## 3. Experimental Setup

### Pruning Method
`torch.nn.utils.prune.random_unstructured` — applied **at initialization**, permanent for the entire run. Pruning percentages sweep 0–100% with fine-grained steps near the transition (82, 84, 86, 88, 92, 94, 96, 98%).

### Architectures & Datasets

| Architecture | Datasets | Status |
|---|---|---|
| SLP (Single-Layer Perceptron) | MNIST, Fashion-MNIST | ✅ Complete |
| SLP | CIFAR-10 | ⚠️ Anomalous — learned only at P%=90–94%; not pursued |
| SLP | CIFAR-100 | ❌ No learning observed; dropped after one sanity run |
| CNN (Convolutional) | MNIST, Fashion-MNIST, CIFAR-10 | ✅ Complete |
| DenseNet | — | 🔲 Out of scope / not started |

### Training Configuration

| Parameter | Value |
|---|---|
| Runs per configuration | 100 |
| Optimizer | Adadelta |
| Loss | Cross-Entropy |
| Stopping criterion | Relative change in mean CE_Test over consecutive 20-batch windows < 1% |
| SLP batch sizes | 64, 1024, 60000 (full-batch) |
| CNN batch sizes | 64, 1024 |
| SLP pruning target | `ALL` layers |
| CNN pruning targets | `CONV`, `FHL`, `SHL`, `FHL+SHL`, `ALL` |

### Normalization
Dataset-specific normalization is used (not fixed 0.5), as it yields lower final CE and higher IPA:
- MNIST: μ=0.1307, σ=0.3081
- Fashion-MNIST: μ=0.2860, σ=0.3530

---

## 4. Repository Structure

```
physlab/
├── SLP/
│   ├── SLP-MNIST/         # Training scripts, analysis notebooks, fitting data
│   ├── SLP-FMNIST/        # Training scripts, analysis notebooks
│   ├── SLP-CIFAR10/       # Anomalous results; not pursued
│   └── SLP-CIFAR100/      # Single sanity run only
├── Convolution/
│   └── Convolutional/
│       ├── Convolutional-MNIST/
│       ├── Convolutional-FMNIST/
│       ├── Convolutional-CIFAR-10/
│       └── Convolutional-CIFAR-100/
├── new_graph/Graphs/       # All publication-ready figures
│   ├── SLP/{SLP-MNIST, SLP-FMNIST}/
│   └── Convolutional/{Convolution-MNIST, Convolutional-FMNIST, Convolutional_CIFAR-10}/
├── paper/                  # Draft paper: Neural research.docx
├── DenseNet/               # Not started (out of scope)
├── new_test/               # Scratch space (ignore)
├── README.md               # Full project documentation
├── FITTING_DATA_LOCATION.md # Data storage guide for fitting pipeline
└── md_file_reader.ipynb    # Utility notebook
```

Raw training output format (`.txt` logs):
```
Current_Epoch | Batch/Total | CE_Train | Accuracy(%) | CE_Test | Batch_Number
```

---

## 5. Data & Analysis Pipeline

### Step 1 — Training
Run experiment scripts from their own directory. They write `.txt` logs into nested directories: `prune_layers_*/p-percentage_*/batch_size_*/`.

### Step 2 — Averaging & AUC
Jupyter notebooks read `.txt` logs and produce:
- Averaged CE / accuracy curves per pruning %
- AUC (area under the CE-vs-batch curve) as an alternative IPA scalar
- Plots committed to `new_graph/Graphs/`

### Step 3 — Curve Fitting (SLP-MNIST, in progress)
Each per-run CE curve is fit to `A + B/(x+1)^n`, where `A = CE_asy`. This is the route to a universal IPA definition that works for CNN too.

Fitting pipeline output (stored in `SLP/SLP-MNIST/`):

| Directory | Contents |
|---|---|
| `Fitting_IPA_data/` | Per-batch-size summary CSVs, per-run parameter CSVs, text log |
| `Fitting_IPA_curves_data/BS_{bs}/` | Pre-computed averaged raw data, mean fit curves, all 100 individual fits |
| `Fitting_IPA_graphs/` | IPA vs P%, BN_L vs P%, CE_L vs P% plots |

---

## 6. Key Findings So Far

- **SLP phase transition near P% ≈ 80%** — sharp drop in IPA observed around this threshold.
- **CNN phase transition near P% ≈ 90%** — consistent across MNIST, Fashion-MNIST, CIFAR-10.
- SLP cannot learn CIFAR-10 in general (anomalous partial learning between 90–94% is unexplained).
- SLP cannot learn CIFAR-100 at all.
- Dataset-specific normalization meaningfully improves learning vs. fixed 0.5 normalization.

---

## 7. In-Progress & Open Problems

| Item | Status | Notes |
|---|---|---|
| Curve-fitting asymptote (`CE_asy` via `A + B/(x+1)^n`) | 🔄 In progress | Working in `SLP-MNIST/fitting_function_IPA.ipynb` and SLP-FMNIST notebooks |
| AUC-based IPA | 🔄 In progress | Alternative scalar; being evaluated alongside fitting-based IPA |
| Batch-size correction factor for IPA | 🔄 In progress | Expected but not yet verified experimentally |
| IPA-vs-P% publication figures | ⏳ Pending | Waiting for asymptote method to stabilize |
| CNN curve-fitting pipeline | ⏳ Pending | Depends on asymptote method stabilizing from SLP work |
| DenseNet experiments | 🔲 Out of scope | Not started |

---

## 8. Out of Scope

- `DenseNet/` — planned future architecture, not started
- `new_test/` — scratch space, ignore
- SLP on CIFAR-10 and CIFAR-100 — anomalous / no learning; excluded from main analysis

---

## 9. Requirements

- Python 3
- PyTorch + torchvision
- CUDA (optional but recommended for CNN runs)
- Jupyter, matplotlib, numpy, scipy (for notebooks and fitting)
