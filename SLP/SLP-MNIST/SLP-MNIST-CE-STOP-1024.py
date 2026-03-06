import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune

# Define a simple fully connected model (SLP) with pruning
class PrunedMLP(nn.Module):
    def __init__(self, prune_percentage=50.0, Prune_Layers='ALL'):
        super(PrunedMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 10, bias=False)
        self.relu = nn.ReLU()
        self.prune_model(prune_percentage, Prune_Layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def prune_model(self, prune_percentage, Prune_Layers):
        for name, module in self.named_modules():
            if Prune_Layers == 'ALL' and isinstance(module, nn.Linear):
                prune.random_unstructured(module, name='weight', amount=prune_percentage)


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


def train(model, train_loader, test_loader, criterion, optimizer, fraction_of_epoch, device, output_file):
    with open(output_file, 'w') as f:
        f.write(f'{"Current_Epoch":<15}{"Batch/Total":<20}{"CE_Train":<20}{"Accuracy(%)":<20}{"CE_TEST":<20}{"Batch_Number":<15}\n')

        test_loss, accuracy = test(model, test_loader, criterion, device=device)
        f.write(f'{"0":<15}{"[0/60000]":<20}{"--":<20}{"9.020":<20}{test_loss:<20.4f}0\n')

        model.train()
        batch_number = 1
        total_batches = len(train_loader)
        target_batches = int(fraction_of_epoch * total_batches)
        max_epoch = 2000

        ce_segments = []
        current_segment = []

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
                            print(f"Stopping early at batch {batch_number} (Avg CE_TEST in last 20 batches within 1% of previous 20).")
                            return

                    current_segment = []

                accuracy_str = f'{accuracy * 100:.4f}'
                f.write(f'{epoch + 1:<15}{f"[{(batch_idx+1) * len(inputs)}/{len(train_loader.dataset)}]":<20}{loss.item():<20.4f}{accuracy_str:<20}{test_loss:<20.4f}{batch_number:<15}\n')
                batch_number += 1


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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    PRUNE_LAYERS_OPTIONS = ['ALL']
    ACCEPTABLE_PRUNE_PERCENTAGES = [i/100 for i in range(0,110,10)]
    ACCEPTABLE_BATCH_SIZES = [1024]
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
                    model = PrunedMLP(prune_percentage, Prune_Layers).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adadelta(model.parameters())

                    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    batch_directory = f"batch_size_{batch_size}"
                    os.makedirs(os.path.join(prune_layers_directory, prune_directory, batch_directory), exist_ok=True)

                    output_file = os.path.join(
                        prune_layers_directory,
                        prune_directory,
                        batch_directory,
                        f"slp_{prune_percentage}_{batch_size}_run_{run_index}.txt"
                    )

                    epochs = calculate_epochs_for_batch_size(batch_size)

                    train(model, train_loader, test_loader, criterion, optimizer,
                          fraction_of_epoch=epochs, device=device, output_file=output_file)


if __name__ == "__main__":
    main()

