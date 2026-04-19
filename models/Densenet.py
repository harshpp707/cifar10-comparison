import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            bn_size * growth_rate,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super(DenseBlock, self).__init__()

        layers = []
        channels = in_channels

        for _ in range(num_layers):
            layers.append(
                DenseLayer(
                    channels,
                    growth_rate,
                    bn_size,
                    drop_rate
                )
            )
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class SimpleDenseNet(nn.Module):

    def __init__(
        self,
        growth_rate=16,
        block_layers=(6, 6, 6),
        num_init_features=32,
        bn_size=4,
        drop_rate=0.2,
        compression=0.5,
        num_classes=10
    ):
        super(SimpleDenseNet, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(
            3,
            num_init_features,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu1 = nn.ReLU(inplace=True)

        num_features = num_init_features
        blocks = []

        # Dense blocks
        for i, layers in enumerate(block_layers):

            dense_block = DenseBlock(
                layers,
                num_features,
                growth_rate,
                bn_size,
                drop_rate
            )

            blocks.append(dense_block)
            num_features = dense_block.out_channels

            if i != len(block_layers) - 1:
                out_features = int(num_features * compression)

                blocks.append(
                    Transition(
                        num_features,
                        out_features
                    )
                )

                num_features = out_features

        self.features = nn.Sequential(*blocks)

        self.bn_final = nn.BatchNorm2d(num_features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(
            num_features,
            num_classes
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.features(x)

        x = F.relu(self.bn_final(x))

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x