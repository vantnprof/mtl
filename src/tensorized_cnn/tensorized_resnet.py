import torch
import torch.nn as nn
# from src.tensorized_cnn.tensorizedconv2d import TensorizedConv2d
from src.mtl import MultilinearMuTransformationLayer as MTL
from src.tensorized_cnn.bottleneckMTL import BottleneckMTL
from src.tensorized_cnn.basicblockMTL import BasicBlockMTL


def print_layer_shapes(model, model_name, input_tensor=None):
    print(f"\n{model_name} Layer Outputs:")
    x = input_tensor
    print(f"Input: {x.shape}")
    x = model.mtl1(x)
    print(f"After conv1: {x.shape}")
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    print(f"After maxpool: {x.shape}")
    x = model.layer1(x)
    print(f"After layer1: {x.shape}")
    x = model.layer2(x)
    print(f"After layer2: {x.shape}")
    x = model.layer3(x)
    print(f"After layer3: {x.shape}")
    x = model.layer4(x)
    print(f"After layer4: {x.shape}")
    x = model.avgpool(x)
    print(f"After avgpool: {x.shape}")
    x = torch.flatten(x, 1)
    print(f"After flatten: {x.shape}")
    x = model.fc(x)
    print(f"After fc: {x.shape}")


class TensorizedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(TensorizedResNet, self).__init__()
        self.in_channels = 64

        self.mtl1 = MTL(
            input_channels=3,
            output_channels=self.in_channels,
            kernel_size=(7, 7),
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                MTL(
                    input_channels=self.in_channels,
                    output_channels=out_channels * block.expansion,
                    kernel_size=(1, 1),
                    stride=stride,
                    padding=0
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MTL):
                if hasattr(m, 'weights') and m.weights is not None:
                    nn.init.kaiming_normal_(m.weights, mode='fan_out', nonlinearity='relu')


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
    
    
def static_mtl_resnet18(num_classes=1000):
    return TensorizedResNet(BasicBlockMTL, [2, 2, 2, 2], num_classes)

def static_mtl_resnet34(num_classes=1000):
    return TensorizedResNet(BasicBlockMTL, [3, 4, 6, 3], num_classes)

def static_mtl_resnet50(num_classes=1000):
    return TensorizedResNet(BottleneckMTL, [3, 4, 6, 3], num_classes)

def static_mtl_resnet101(num_classes=1000):
    return TensorizedResNet(BottleneckMTL, [3, 4, 23, 3], num_classes)

def static_mtl_resnet152(num_classes=1000):
    return TensorizedResNet(BottleneckMTL, [3, 8, 36, 3], num_classes)
    

if __name__ == "__main__":
    # print out the shapes of tensors after each layer
    batch_size = 32
    num_classes = 10
    from torchvision import models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device: ", device)

    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    torchvsion_resnet18 = models.resnet18(weights=None, num_classes=num_classes).to(device)
    # print_layer_shapes(torchvsion_resnet18, "torchvision resnet18", input_tensor)
    print("Number of parameters in torchvision resnet18: ", sum(p.numel() for p in torchvsion_resnet18.parameters()))
    tensorized_resnet18_model = static_mtl_resnet18(num_classes=num_classes).to(device)
    # print_layer_shapes(tensorized_resnet18_model, "tensorized resnet18", input_tensor)
    print("Number of parameters in tensorized resnet18: ", sum(p.numel() for p in tensorized_resnet18_model.parameters()))
    print("Saving ratio: ", sum(p.numel() for p in torchvsion_resnet18.parameters()) / sum(p.numel() for p in tensorized_resnet18_model.parameters()))
    print("#" * 50)
    torchvsion_resnet34 = models.resnet34(weights=None, num_classes=num_classes).to(device)
    # print_layer_shapes(torchvsion_resnet34, "torchvision resnet34", input_tensor)
    print("Number of parameters in torchvision resnet34: ", sum(p.numel() for p in torchvsion_resnet34.parameters()))
    tensorized_resnet34_model = static_mtl_resnet34(num_classes=num_classes).to(device)
    # print_layer_shapes(tensorized_resnet34_model, "tensorized resnet34", input_tensor)
    print("Number of parameters in tensorized resnet34: ", sum(p.numel() for p in tensorized_resnet34_model.parameters()))
    print("Saving ratio: ", sum(p.numel() for p in torchvsion_resnet34.parameters()) / sum(p.numel() for p in tensorized_resnet34_model.parameters()))
    print("#" * 50)

    torchvsion_resnet50 = models.resnet50(weights=None, num_classes=num_classes).to(device)
    # print_layer_shapes(torchvsion_resnet50, "torchvision resnet50", input_tensor)
    print("Number of parameters in torchvision resnet50: ", sum(p.numel() for p in torchvsion_resnet50.parameters()))
    tensorized_resnet50_model = static_mtl_resnet50(num_classes=num_classes).to(device)
    # print_layer_shapes(tensorized_resnet50_model, "tensorized resnet50", input_tensor)
    print("Number of parameters in tensorized resnet50: ", sum(p.numel() for p in tensorized_resnet50_model.parameters()))
    print("Saving ratio: ", sum(p.numel() for p in torchvsion_resnet50.parameters()) / sum(p.numel() for p in tensorized_resnet50_model.parameters()))
    print("#" * 50)

    torchvsion_resnet101 = models.resnet101(weights=None, num_classes=num_classes).to(device)
    # print_layer_shapes(torchvsion_resnet101, "torchvision resnet101", input_tensor)
    print("Number of parameters in torchvision resnet101: ", sum(p.numel() for p in torchvsion_resnet101.parameters()))
    tensorized_resnet101_model = static_mtl_resnet101(num_classes=num_classes).to(device)
    # print_layer_shapes(tensorized_resnet101_model, "tensorized resnet101", input_tensor)
    print("Number of parameters in tensorized resnet101: ", sum(p.numel() for p in tensorized_resnet101_model.parameters()))
    print("Saving ratio: ", sum(p.numel() for p in torchvsion_resnet101.parameters()) / sum(p.numel() for p in tensorized_resnet101_model.parameters()))
    print("#" * 50)

    torchvsion_resnet152 = models.resnet152(weights=None, num_classes=num_classes).to(device)
    # print_layer_shapes(torchvsion_resnet152, "torchvision resnet152", input_tensor)
    print("Number of parameters in torchvision resnet152: ", sum(p.numel() for p in torchvsion_resnet152.parameters()))
    tensorized_resnet152_model = static_mtl_resnet152(num_classes=num_classes).to(device)
    # print_layer_shapes(tensorized_resnet152_model, "tensorized resnet152", input_tensor)
    print("Number of parameters in tensorized resnet152: ", sum(p.numel() for p in tensorized_resnet152_model.parameters()))
    print("Saving ratio: ", sum(p.numel() for p in torchvsion_resnet152.parameters()) / sum(p.numel() for p in tensorized_resnet152_model.parameters()))
    print("#" * 50)


    # measure the time taken to run the model
    import time
    import numpy as np
    from tqdm import tqdm
    N = 100  # Number of repetitions for timing

    model_timings = []

    models_to_test = [
        ("resnet18", tensorized_resnet18_model, torchvsion_resnet18),
        ("resnet34", tensorized_resnet34_model, torchvsion_resnet34),
        ("resnet50", tensorized_resnet50_model, torchvsion_resnet50),
        ("resnet101", tensorized_resnet101_model, torchvsion_resnet101),
        ("resnet152", tensorized_resnet152_model, torchvsion_resnet152),
    ]

    for name, tensorized_model, torchvision_model in models_to_test:
        tensorized_times = []
        torchvision_times = []
        print(f"\nTiming {name} model:")
        for _ in tqdm(range(N)):
            # Tensorized
            start = time.time()
            with torch.no_grad():
                _ = tensorized_model(input_tensor)
            tensorized_times.append(time.time() - start)

            # Torchvision
            start = time.time()
            with torch.no_grad():
                _ = torchvision_model(input_tensor)
            torchvision_times.append(time.time() - start)

        tensorized_times = np.array(tensorized_times)
        torchvision_times = np.array(torchvision_times)
        model_timings.append((
            name,
            tensorized_times.mean(), tensorized_times.std(),
            torchvision_times.mean(), torchvision_times.std()
        ))

    # Print as a comparison table
    print("\n" + "#" * 80)
    print(f"{'Model':<12} | {'Tensorized Mean (s)':>20} | {'Tensorized Std (s)':>18} | {'Torchvision Mean (s)':>20} | {'Torchvision Std (s)':>18}")
    print("-" * 80)
    for name, t_mean, t_std, v_mean, v_std in model_timings:
        print(f"{name:<12} | {t_mean:20.6f} | {t_std:18.6f} | {v_mean:20.6f} | {v_std:18.6f}")
    print("#" * 80)