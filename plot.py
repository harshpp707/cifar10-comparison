import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------------------------------
# 1. Import your custom model
# -------------------------------
sys.path.append("./models")  # adjust if needed
from cnn import SimpleCNN


# -------------------------------
# 2. Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# 3. Load model + weights
# -------------------------------
model = SimpleCNN()  # make sure args match training
model.load_state_dict(torch.load("results/cnn_model.pth", map_location=device))
model.to(device)
model.eval()


# -------------------------------
# 4. Load CIFAR-10 test set
# -------------------------------
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# -------------------------------
# 5. Run inference
# -------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# -------------------------------
# 6. Confusion Matrix
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)
classes = test_dataset.classes

plt.figure(figsize=(10, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("CNNConfusion.png")
plt.show()


# -------------------------------
# 7. Per-class accuracy
# -------------------------------
cm = np.array(cm)
class_acc = cm.diagonal() / cm.sum(axis=1)

print("\nPer-class accuracy:")
for i, acc in enumerate(class_acc):
    print(f"{classes[i]}: {acc:.4f}")


# -------------------------------
# 8. Overall accuracy
# -------------------------------
overall_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\nOverall Accuracy: {overall_acc:.4f}")