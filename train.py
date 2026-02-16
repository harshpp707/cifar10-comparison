import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import os

from models.cnn import SimpleCNN
from utils import get_device, plot_curves

from google.colab import drive
drive.mount('/content/drive')
drive_folder = "/content/drive/MyDrive/cifar_results"
os.makedirs(drive_folder, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

def main():
    device = get_device()
    print("Using device:", device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train)

    train_size = 45000
    val_size = 5000
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 20

    train_losses, val_losses = [], []
    train_acc, val_acc = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_acc.append(tr_acc)
        val_acc.append(val_accuracy)

        print(f"Train Acc: {tr_acc:.2f}% | Val Acc: {val_accuracy:.2f}%")

    model_path = os.path.join(drive_folder, "cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plot_curves(train_losses, val_losses, train_acc, val_acc, drive_folder)
    print(f"Plots saved to {drive_folder}")

    print("\nEvaluating on Test Set")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
