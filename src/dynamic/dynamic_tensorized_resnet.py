import torch
import torch.nn as nn
import torch.profiler
from src.dynamic.dynamic_mtl import DynamicMTL
from src.dynamic.dynamic_basicblockMTL import DynamicBasicBlockMTL
from src.dynamic.dynamic_bottleneckmtl import DynamicBottleNeckMTL
from typing import Tuple
from src.dynamic.utils import find_config


class DynamicTensorizedResNet(nn.Module):
    def __init__(self, block, layers, max_features: Tuple[int]=None, max_rank: int=None, num_classes: int = 10):
        super(DynamicTensorizedResNet, self).__init__()
        # max_features: (channels, height, width) for the first layer
        self.in_channels = max_features[0]
        self.mtl1 = DynamicMTL(
            input_channels=3,
            output_channels=self.in_channels,
            target_params=7**2*3*self.in_channels
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # For simplicity, keep spatial size the same for all layers (or you can update as needed)
        self.layer1 = self._make_layer(block, 64, layers[0], max_features, max_rank, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], max_features, max_rank, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], max_features, max_rank, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], max_features, max_rank, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, max_features, max_rank, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                DynamicMTL(
                    input_channels=self.in_channels,
                    output_channels=out_channels * block.expansion,
                    target_params=1**2*self.in_channels*out_channels * block.expansion,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, downsample=downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
                pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.relu(self.bn1(self.mtl1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Example constructor
# def dynamic_mtl_resnet18(max_features=(64, 56, 56), max_rank=100, num_classes=10):
#     return DynamicTensorizedResNet(DynamicBasicBlockMTL, [2, 2, 2, 2], max_features, max_rank, num_classes)

# def dynamic_mtl_resnet34(max_features=(64, 56, 56), max_rank=100, num_classes=10):
#     return DynamicTensorizedResNet(DynamicBasicBlockMTL, [3, 4, 6, 3], max_features, max_rank, num_classes)

# def dynamic_tensorized_resnet50(max_features=(64, 56, 56), max_rank=100, num_classes=10):
#     return DynamicTensorizedResNet(DynamicBottleNeckMTL, [3, 4, 6, 3], max_features, max_rank, num_classes)

# def dynamic_tensorized_resnet101(max_features=(64, 56, 56), max_rank=100, num_classes=10):
#     return DynamicTensorizedResNet(DynamicBottleNeckMTL, [3, 4, 23, 3], max_features, max_rank, num_classes)

# def dynamic_tensorized_resnet152(max_features=(64, 56, 56), max_rank=100, num_classes=10):
#     return DynamicTensorizedResNet(DynamicBottleNeckMTL, [3, 8, 36, 3], max_features, max_rank, num_classes)


if __name__ == "__main__":
    import time
    import torch.cuda
    from torchvision import models

    max_features = (128, 128, 128)
    max_rank = 100
    num_classes = 100
    N = 10

    input_tensor = torch.randn(10, 3, 224, 224, device="cuda", dtype=torch.float32).contiguous(memory_format=torch.channels_last)

    print("==== TorchVision ResNet-18 ====")
    torchvision_model = models.resnet18(weights=None, num_classes=num_classes).to("cuda", memory_format=torch.channels_last)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        start_time = time.time()
        for _ in range(N):
            output = torchvision_model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
    print(f"Inference Time: {(end_time - start_time) / N:.6f} s")

    print("==== DynamicTensorizedResNet-18 ====")
    model = dynamic_mtl_resnet18(max_features=max_features, max_rank=max_rank, num_classes=num_classes).to("cuda", memory_format=torch.channels_last)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        start_time = time.time()
        for _ in range(N):
            output = model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
    print(f"Inference Time: {(end_time - start_time) / N:.6f} s")

    print("==== Backward Pass Timing ====")
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, num_classes, (input_tensor.size(0),), device="cuda")

    torchvision_model.train()
    optimizer = torch.optim.SGD(torchvision_model.parameters(), lr=0.01)
    start_time = time.time()
    for _ in range(N):
        optimizer.zero_grad()
        out = torchvision_model(input_tensor)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"TorchVision ResNet-18 Backward Time: {(end_time - start_time) / N:.6f} s")

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    start_time = time.time()
    for _ in range(N):
        optimizer.zero_grad()
        out = model(input_tensor)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"DynamicTensorizedResNet-18 Backward Time: {(end_time - start_time) / N:.6f} s")

    print("\n==== CUDA Memory Stats ====")
    print(torch.cuda.memory_summary())