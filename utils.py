import torch
import matplotlib.pyplot as plt
import os


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_curves(train_losses, val_losses, train_acc, val_acc):
    os.makedirs("results", exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("results/loss_curve.png")

    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig("results/accuracy_curve.png")
