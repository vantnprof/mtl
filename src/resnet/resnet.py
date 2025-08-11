import torch
import torch.nn as nn
import torch.nn.functional as F
from src.resnet.basicblock import BasicBlock
from src.resnet.bottleneck import Bottleneck


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, weights=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self._weights = weights

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
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
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        if self._weights:
            self.load_state_dict(torch.load(self._weights))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BatchNorm in each residual branch
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=10, weights=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, weights)

def resnet34(num_classes=10, weights=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, weights)

def resnet50(num_classes=10, weights=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, weights)

def resnet101(num_classes=10, weights=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, weights)

def resnet152(num_classes=10, weights=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, weights)


# Define the ResNet-18 Model
# class ResNet18(nn.Module):
#     def __init__(self, block=BasicBlock, layers=[2,2,2,2], weights=None, num_classes=10):
#         super(ResNet18, self).__init__()
#         self.in_channels = 64

#         # Initial convolutional layer
#         self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
#                                stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         # Max pooling layer
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # Residual layers
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # Adaptive average pooling and fully connected layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         # Initialize weights
#         self._weights = weights
#         self._initialize_weights()

#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         # Determine if downsampling is needed
#         if stride != 1 or self.in_channels != out_channels * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, out_channels * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * block.expansion),
#             )

#         layers = []
#         # First block may involve downsampling
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels * block.expansion
#         # Remaining blocks
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))

#         return nn.Sequential(*layers)

#     def _initialize_weights(self):
#         if self._weights:
#             # Load weights if provided
#             self.load_state_dict(torch.load(self._weights))
#         else:
#             # Initialize weights using Kaiming He initialization
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out',
#                                             nonlinearity='relu')
#                 elif isinstance(m, nn.BatchNorm2d):
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Forward pass through initial layers
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         # Forward pass through residual layers
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         # Global average pooling and classification
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x
    

if __name__ == '__main__':
    # Example usage
    # model = ResNet18(weights=None, num_classes=10)
    num_classes = 1000
    resnet18_model = resnet18(num_classes=num_classes, weights=None)
    resnet18_model = resnet18_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    resnet34_model = resnet34(num_classes=num_classes, weights=None)
    resnet34_model = resnet34_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50_model = resnet50(num_classes=num_classes, weights=None)
    resnet50_model = resnet50_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    resnet101_model = resnet101(num_classes=num_classes, weights=None)   
    resnet101_model = resnet101_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    resnet152_model = resnet152(num_classes=num_classes, weights=None)
    resnet152_model = resnet152_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test the model with random input
    x = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels (RGB), 224x224 image
    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')

    resnet18_output = resnet18_model(x)
    print(f"ResNet-18 output shape: {resnet18_output.shape}")
    resnet34_output = resnet34_model(x)
    print(f"ResNet-34 output shape: {resnet34_output.shape}")
    resnet50_output = resnet50_model(x)
    print(f"ResNet-50 output shape: {resnet50_output.shape}")
    resnet101_output = resnet101_model(x)
    print(f"ResNet-101 output shape: {resnet101_output.shape}")
    resnet152_output = resnet152_model(x)
    print(f"ResNet-152 output shape: {resnet152_output.shape}")
    # Print the model summary

    # Count the number of parameters in the model
    resnet18_num_params = sum(p.numel() for p in resnet18_model.parameters() if p.requires_grad)
    print(f"Number of parameters in ResNet-18: {resnet18_num_params}")
    resnet34_num_params = sum(p.numel() for p in resnet34_model.parameters() if p.requires_grad)
    print(f"Number of parameters in ResNet-34: {resnet34_num_params}")
    resnet50_num_params = sum(p.numel() for p in resnet50_model.parameters() if p.requires_grad)
    print(f"Number of parameters in ResNet-50: {resnet50_num_params}")
    resnet101_num_params = sum(p.numel() for p in resnet101_model.parameters() if p.requires_grad)
    print(f"Number of parameters in ResNet-101: {resnet101_num_params}")
    resnet152_num_params = sum(p.numel() for p in resnet152_model.parameters() if p.requires_grad)
    print(f"Number of parameters in ResNet-152: {resnet152_num_params}")   


    from torchvision import models
    torchvsion_resnet18 = models.resnet18(weights=None)
    torchvsion_resnet34 = models.resnet34(weights=None)
    torchvsion_resnet50 = models.resnet50(weights=None)
    torchvsion_resnet101 = models.resnet101(weights=None)
    torchvsion_resnet152 = models.resnet152(weights=None)
    torchvsion_resnet18 = torchvsion_resnet18.to('cuda' if torch.cuda.is_available() else 'cpu')
    torchvsion_resnet34 = torchvsion_resnet34.to('cuda' if torch.cuda.is_available() else 'cpu')
    torchvsion_resnet50 = torchvsion_resnet50.to('cuda' if torch.cuda.is_available() else 'cpu')
    torchvsion_resnet101 = torchvsion_resnet101.to('cuda' if torch.cuda.is_available() else 'cpu')
    torchvsion_resnet152 = torchvsion_resnet152.to('cuda' if torch.cuda.is_available() else 'cpu')
    torchvsion_resnet18_num_params = sum(p.numel() for p in torchvsion_resnet18.parameters() if p.requires_grad)
    print(f"Number of parameters in Torchvision ResNet-18: {torchvsion_resnet18_num_params}")
    torchvsion_resnet34_num_params = sum(p.numel() for p in torchvsion_resnet34.parameters() if p.requires_grad)
    print(f"Number of parameters in Torchvision ResNet-34: {torchvsion_resnet34_num_params}")
    torchvsion_resnet50_num_params = sum(p.numel() for p in torchvsion_resnet50.parameters() if p.requires_grad)
    print(f"Number of parameters in Torchvision ResNet-50: {torchvsion_resnet50_num_params}")
    torchvsion_resnet101_num_params = sum(p.numel() for p in torchvsion_resnet101.parameters() if p.requires_grad)
    print(f"Number of parameters in Torchvision ResNet-101: {torchvsion_resnet101_num_params}")
    torchvsion_resnet152_num_params = sum(p.numel() for p in torchvsion_resnet152.parameters() if p.requires_grad)
    print(f"Number of parameters in Torchvision ResNet-152: {torchvsion_resnet152_num_params}")