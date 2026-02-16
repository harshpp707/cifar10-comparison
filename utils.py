import torch
import matplotlib.pyplot as plt
import os

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_curves(train_losses, val_losses, train_acc, val_acc, drive_folder):
    os.makedirs(drive_folder, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{drive_folder}/loss_curve.png")

    plt.figure()
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{drive_folder}/accuracy_curve.png")
