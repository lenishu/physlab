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
        # Flatten the 32x32x3 input images (CIFAR-10)
        self.flatten = nn.Flatten()
        # Define a single fully connected layer (input to output)
        self.fc1 = nn.Linear(3 * 32 * 32, 10, bias=False)  # Input to output (10 classes for CIFAR-10)
        self.relu = nn.ReLU()

        # Prune the model based on the specified layers
        self.prune_model(prune_percentage, Prune_Layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def prune_model(self, prune_percentage, Prune_Layers):
        for name, module in self.named_modules():
            if Prune_Layers == 'ALL' and isinstance(module, nn.Linear):
                prune.random_unstructured(module, name='weight', amount=prune_percentage)

def train(model, train_loader, test_loader, criterion, optimizer, fraction_of_epoch, device, output_file):
    with open(output_file, 'w') as f:
        f.write(f'{"Current_Epoch":<15}{"Batch/Total":<20}{"CE_Train":<20}{"Accuracy(%)":<20}{"CE_TEST":<20}{"Batch_Number":<15}\n')

        # Test at the very beginning (epoch 0, batch 0)
        test_loss, accuracy = test(model, test_loader, criterion, device=device)
        f.write(f'{"0":<15}{"[0/50000]":<20}{"--":<20}{"9.020":<20}{test_loss:<20.4f}0\n')

        model.train()
        batch_number = 1
        total_batches = len(train_loader)  # Total number of batches in one epoch

        # Calculate how many batches we should process based on fraction of epoch
        target_batches = int(fraction_of_epoch * total_batches)

        for epoch in range(100):  # We are allowing up to 100 epochs, but will stop earlier if target_batches are processed
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if batch_number > target_batches:
                    print(f"Stopping after processing {batch_number} batches (fraction of epoch completed).")
                    return  # Stop training when target batch number is reached

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                test_loss, accuracy = test(model, test_loader, criterion, device=device)
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
    """
    Calculate number of epochs based on the batch size.
    """
    if batch_size == 64:
        return 1
    elif batch_size == 256:
        return 1
    elif batch_size == 4000:
        return 10
    elif batch_size == 50000:
        return 100
    else:
        raise ValueError("Unsupported batch size")

def main():
    CUDA = True
    device = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 dataset and create DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    PRUNE_LAYERS_OPTIONS = ['ALL']
    ACCEPTABLE_PRUNE_PERCENTAGES = [i / 100 for i in range(0, 110, 10)]
    ACCEPTABLE_BATCH_SIZES = [64, 50000]

    num_runs = 100

    for run_index in range(num_runs):
        for prune_layers in PRUNE_LAYERS_OPTIONS:
            # Create directory for each prune layers option
            prune_layers_directory = f"prune_layers_{prune_layers}"
            os.makedirs(prune_layers_directory, exist_ok=True)

            for prune_percentage in ACCEPTABLE_PRUNE_PERCENTAGES:
                # Create directory for each pruning percentage
                prune_directory = f"p-percentage_{prune_percentage}"
                os.makedirs(os.path.join(prune_layers_directory, prune_directory), exist_ok=True)

                for batch_size in ACCEPTABLE_BATCH_SIZES:
                    Prune_Layers = prune_layers  # Change this variable to control pruning in different layers
                    model = PrunedMLP(prune_percentage, Prune_Layers).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adadelta(model.parameters())

                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    # Create subdirectory for each batch size within the pruning percentage directory
                    batch_directory = f"batch_size_{batch_size}"
                    os.makedirs(os.path.join(prune_layers_directory, prune_directory, batch_directory), exist_ok=True)

                    # Set output file path with run index
                    output_file = os.path.join(prune_layers_directory, prune_directory, batch_directory, f"slp__{prune_percentage}_{batch_size}_run_{run_index}.txt")

                    # Calculate the number of epochs dynamically based on the batch size
                    epochs = calculate_epochs_for_batch_size(batch_size)

                    # Train the model and save output file
                    train(model, train_loader, test_loader, criterion, optimizer, fraction_of_epoch=epochs, device=device, output_file=output_file)

if __name__ == "__main__":
    main()

