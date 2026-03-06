# PhysLab - Neural Network Pruning Research

A comprehensive research project investigating the effects of neural network pruning on learning performance across various architectures and datasets.

## Overview

This project explores how **weight pruning** affects the learning capabilities of neural networks. Experiments are conducted on:

- **Single Layer Perceptrons (SLP)** - Simple fully connected networks
- **Convolutional Neural Networks (CNN)** - Deep learning models with convolutional layers

## Project Structure

```
physlab/
├── SLP/                          # Single Layer Perceptron experiments
│   ├── SLP-MNIST/               # MNIST digit classification
│   ├── SLP-FMNIST/              # Fashion-MNIST classification
│   ├── SLP-CIFAR10/             # CIFAR-10 image classification
│   └── SLP-CIFAR100/            # CIFAR-100 image classification
│
└── Convolution/                  # Convolutional network experiments
    ├── Conv-FMNIST/             # Fashion-MNIST with CNNs
    └── Convolutional/           # Multi-dataset CNN experiments
        ├── Convolutional-MNIST/
        ├── Convolutional-FMNIST/
        ├── Convolutional-CIFAR-10/
        └── Convolutional-CIFAR-100/
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- CUDA (optional, for GPU acceleration)

## Experimental Setup

### Pruning Methodology

- **Pruning Type**: Random unstructured pruning on weight matrices
- **Pruning Percentages**: 0%, 10%, 20%, ..., 100%
- **Batch Sizes**: 64, 1024, 60000 (full-batch)
- **Optimizer**: Adadelta
- **Loss Function**: Cross-Entropy (CE)

### Stopping Criteria

Training stops when the average Cross-Entropy (CE) over 20 consecutive batches changes by less than 1% relative to the previous 20-batch average.

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| **CE_Train** | Cross-entropy loss on training data |
| **CE_Test** | Cross-entropy loss on test data |
| **Accuracy** | Classification accuracy (%) |
| **IPA** | Information Processing Accuracy |

## Key Findings

### SLP Experiments

- ✅ **MNIST**: SLP successfully learns with various pruning levels
- ✅ **Fashion-MNIST**: Similar learning behavior to MNIST
- ⚠️ **CIFAR-10**: Limited learning; only learns with P% between 90-94%
- ❌ **CIFAR-100**: Unable to learn (expected due to complexity)

> **Note**: Normalization using dataset-specific parameters (mean=0.1307, std=0.3081 for MNIST) yields better CE and higher IPA than using fixed 0.5 normalization.

### Convolutional Experiments

- ✅ **MNIST & FMNIST**: CNN successfully learns both datasets
- Extended "longrun" experiments show continued CE decrease with more epochs

## Usage

### Running an SLP Experiment

```bash
cd SLP/SLP-MNIST
python SLP-MNIST-CE-STOP.py
```

### Configuration

Modify the following parameters in the training scripts:

```python
PRUNE_LAYERS_OPTIONS = ['ALL']
ACCEPTABLE_PRUNE_PERCENTAGES = [i/100 for i in range(0, 110, 10)]
ACCEPTABLE_BATCH_SIZES = [64, 60000]
num_runs = 100
```

## Output Format

Results are saved as `.txt` files with columns:

```
Current_Epoch | Batch/Total | CE_Train | Accuracy(%) | CE_Test | Batch_Number
```

## Analysis Tools

Several Jupyter notebooks are provided for visualization:

- `v_1_experiement_ploting_single_plot.ipynb` - Single experiment visualization
- `v_2_creating_all_plot_for_all_batch.ipynb` - Batch-wise analysis
- `v_3_average_of_all_the_run.ipynb` - Averaging across runs
- `v_4_Combined_experiment_avg_and_plot.ipynb` - Combined analysis

Python scripts for IPA analysis:
- `IPA_Analysis_SLP-MNIST_part_I.py`
- `IPA_Analysis_SLP-MNIST_part_II.py`
- `CE_vs_pruning_I.py` / `CE_vs_pruning_II.py`

## License

This project is for academic research purposes.

## Authors

Neural Research Lab
