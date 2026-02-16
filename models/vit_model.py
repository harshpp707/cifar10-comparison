import torch
import torch.nn as nn
import timm

class ViT_CIFAR(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ViT_CIFAR, self).__init__()
        self.model = timm.create_model(
            'vit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
