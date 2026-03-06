import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune

# Define a simple convolutional neural network with pruning
class PrunedConvol(nn.Module):
    def __init__(self, prune_percentage=50.0, Prune_Layers='ALL'):
        super(PrunedConvol, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 28 * 28, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)

        # Prune the model based on the specified layers
        self.prune_model(prune_percentage, Prune_Layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def prune_model(self, prune_percentage, Prune_Layers):
        for name, module in self.named_modules():
            if Prune_Layers == 'CONV' and isinstance(module, nn.Conv2d):
                prune.random_unstructured(module, name='weight', amount=prune_percentage)
            elif Prune_Layers == 'FHL' and isinstance(module, nn.Linear) and 'fc1' in name:
                prune.random_unstructured(module, name='weight', amount=prune_percentage)
            elif Prune_Layers == 'SHL' and isinstance(module, nn.Linear) and 'fc2' in name:
                prune.random_unstructured(module, name='weight', amount=prune_percentage)
            elif Prune_Layers == 'FHL+SHL' and isinstance(module, nn.Linear):
                prune.random_unstructured(module, name='weight', amount=prune_percentage)
            elif Prune_Layers == 'ALL' and isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.random_unstructured(module, name='weight', amount=prune_percentage)

def train(model, train_loader, test_loader, criterion, optimizer, fraction_of_epoch, device, output_file):
    with open(output_file, 'w') as f:
        f.write(f'{"Current_Epoch":<15}{"Batch/Total":<20}{"CE_Train":<20}{"Accuracy(%)":<20}{"CE_TEST":<20}{"Batch_Number":<15}\n')

        test_loss, accuracy = test(model, test_loader, criterion, device=device)
        f.write(f'{"0":<15}{"[0/60000]":<20}{"--":<20}{"9.020":<20}{test_loss:<20.4f}0\n')

        model.train()
        batch_number = 1
        total_batches = len(train_loader)
        target_batches = int(fraction_of_epoch * total_batches)

        current_segment = []
        ce_segments = []

        max_epoch = 2000
        for epoch in range(max_epoch):
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_number > target_batches:
                    print(f"Stopping after processing {batch_number} batches (fraction of epoch completed).")
                    return

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                test_loss, accuracy = test(model, test_loader, criterion, device=device)
                current_segment.append(test_loss)

                if len(current_segment) == 20:
                    current_avg_ce = sum(current_segment) / 20.0
                    ce_segments.append(current_avg_ce)

                    if len(ce_segments) >= 2:
                        prev_avg = ce_segments[-2]
                        relative_change = abs(current_avg_ce - prev_avg) / prev_avg
                        if relative_change <= 0.01:
                            print(f"Early stopping triggered at batch {batch_number} (Avg CE_TEST in last 20 batches within 1% of previous 20).")
                            return
                    current_segment = []

                accuracy_str = f'{accuracy * 100:.4f}'
                f.write(f'{epoch + 1:<15}{f"[{(batch_idx+1) * len(inputs)}/{len(train_loader.dataset)}]":<20}{loss.item():<20.4f}{accuracy_str:<20}{test_loss:<20.4f}{batch_number:<15}\n')
                batch_number += 1

def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= total
    accuracy = correct / total
    return test_loss, accuracy

def calculate_epochs_for_batch_size(batch_size):
    if batch_size == 64:
        return 5
    elif batch_size == 1024:
        return 20
    elif batch_size == 4000:
        return 10
    elif batch_size == 60000:
        return 2000
    else:
        raise ValueError("Unsupported batch size")

def main():
    CUDA = True
    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

    # Updated transform for FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    # Load FashionMNIST instead of MNIST
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

    PRUNE_LAYERS_OPTIONS = ['CONV', 'FHL', 'SHL', 'FHL+SHL', 'ALL']
    ACCEPTABLE_PRUNE_PERCENTAGES = [i / 100 for i in range(90, 100, 2)]
    ACCEPTABLE_BATCH_SIZES = [64, 1024]

    num_runs = 100

    for run_index in range(num_runs):
        for prune_layers in PRUNE_LAYERS_OPTIONS:
            prune_layers_directory = f"prune_layers_{prune_layers}"
            os.makedirs(prune_layers_directory, exist_ok=True)

            for prune_percentage in ACCEPTABLE_PRUNE_PERCENTAGES:
                prune_directory = f"p-percentage_{prune_percentage}"
                os.makedirs(os.path.join(prune_layers_directory, prune_directory), exist_ok=True)

                for batch_size in ACCEPTABLE_BATCH_SIZES:
                    Prune_Layers = prune_layers
                    model = PrunedConvol(prune_percentage, Prune_Layers).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adadelta(model.parameters())

                    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                    batch_directory = f"batch_size_{batch_size}"
                    os.makedirs(os.path.join(prune_layers_directory, prune_directory, batch_directory), exist_ok=True)

                    output_file = os.path.join(
                        prune_layers_directory,
                        prune_directory,
                        batch_directory,
                        f"convol_{prune_percentage}_{batch_size}_run_{run_index}.txt"
                    )

                    epochs = calculate_epochs_for_batch_size(batch_size)
                    train(model, train_loader, test_loader, criterion, optimizer, fraction_of_epoch=epochs, device=device, output_file=output_file)

if __name__ == "__main__":
    main()

