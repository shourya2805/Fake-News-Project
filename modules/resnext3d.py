import torch
import torch.nn as nn


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super().__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class ResNeXt3D(nn.Module):
    def __init__(self, layers=(3, 4, 23, 3), cardinality=32, num_classes=400):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(128, layers[0], cardinality)
        self.layer2 = self._make_layer(256, layers[1], cardinality, stride=2)
        self.layer3 = self._make_layer(512, layers[2], cardinality, stride=2)
        self.layer4 = self._make_layer(1024, layers[3], cardinality, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(cardinality * 32 * ResNeXtBottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * ResNeXtBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * ResNeXtBottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * ResNeXtBottleneck.expansion),
            )
        layers = [ResNeXtBottleneck(self.inplanes, planes, cardinality, stride, downsample)]
        self.inplanes = planes * ResNeXtBottleneck.expansion
        layers += [ResNeXtBottleneck(self.inplanes, planes, cardinality) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.features(x))


def resnext101_kinetics(checkpoint_path):
    model = ResNeXt3D()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("arch") != "resnext-101":
        raise ValueError(f"unexpected checkpoint arch {checkpoint.get('arch')!r}")
    state = {k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model
