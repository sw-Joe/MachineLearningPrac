import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18 and ResNet-34
    This block consists of two 3x3 convolutional layers
    with batch normalization and ReLU activation
    """
    expansion = 1  # Output channels multiplier
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample layer for matching dimensions
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        # Store the input for the skip connection
        identity = x
        
        # First conv block: conv -> batch norm -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv block: conv -> batch norm (no ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling to identity if needed
        # This handles cases where input and output dimensions differ
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # The key ResNet operation: add the skip connection
        out += identity
        
        # Apply ReLU after the addition
        out = F.relu(out)
        
        return out
    

# class ResNet(nn.Module):
#     """
#     Complete ResNet architecture
#     Can be configured for different depths (18, 34, 50, 101, 152)
#     """
    
#     def __init__(self, block, layers, num_classes=1000):
#         super(ResNet, self).__init__()
        
#         # Initial number of channels
#         self.inplanes = 64
        
#         # Initial convolutional layer
#         # Converts 3-channel input to 64 channels
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
#                               padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
        
#         # Max pooling to reduce spatial dimensions
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # Four groups of residual blocks
#         # Each group handles a different feature resolution
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
#         # Global average pooling and final classifier
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
        
#         # Initialize weights properly
#         self._initialize_weights()
    
#     def _make_layer(self, block, planes, blocks, stride=1):
#         """
#         Create a group of residual blocks
        
#         Args:
#             block: The type of residual block to use
#             planes: Number of output channels
#             blocks: Number of blocks in this layer
#             stride: Stride for the first block (for downsampling)
#         """
#         downsample = None
        
#         # Create downsampling layer if needed
#         # This happens when we change the number of channels or spatial size
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                          kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
        
#         layers = []
        
#         # First block might need downsampling
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
        
#         # Remaining blocks don't need downsampling
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
        
#         return nn.Sequential(*layers)
    
#     def _initialize_weights(self):
#         """Initialize network weights using proper initialization schemes"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # He initialization for convolutional layers
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', 
#                                       nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 # Initialize batch norm parameters
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         # Initial processing
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         # Pass through residual block groups
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         # Final classification
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
        
#         return x


# building_block.py 수정 제안

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10): # CIFAR-10은 기본 10개 클래스
        super(ResNet, self).__init__()
        self.inplanes = 16 # CIFAR-10 논문은 첫 채널을 16으로 시작
        
        # [수정] ImageNet용 7x7 stride 2 대신 3x3 stride 1 사용
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # [수정] MaxPool 생략 (이미지 크기가 이미 32x32로 작음)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 논문 구조: 32x32(16), 16x16(32), 8x8(64)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # CIFAR-10용 기본 ResNet은 layer 4개 대신 3개 그룹을 사용하기도 하지만, 
        # 제공된 코드의 4개 그룹을 유지하려면 planes를 [16, 32, 64, 128] 등으로 조정하십시오.

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        self._initialize_weights()


    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a group of residual blocks
        
        Args:
            block: The type of residual block to use
            planes: Number of output channels
            blocks: Number of blocks in this layer
            stride: Stride for the first block (for downsampling)
        """
        downsample = None
        
        # Create downsampling layer if needed
        # This happens when we change the number of channels or spatial size
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        
        # First block might need downsampling
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks don't need downsampling
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using proper initialization schemes"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize batch norm parameters
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # 생략

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x) # 레이어 구성에 따라 선택

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Factory functions for different ResNet variants
# def resnet18(num_classes=1000):
#     """ResNet-18: 2+2+2+2 = 8 residual blocks + initial layers"""
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# def resnet34(num_classes=1000):
#     """ResNet-34: 3+4+6+3 = 16 residual blocks + initial layers"""
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

class PlainBlock(nn.Module):
    """지름길 연결이 없는 일반 컨볼루션 블록"""
    def __init__(self, in_planes, planes, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        # Shortcut 없이 순차적으로 연산 수행
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        return out

class PlainNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PlainNet, self).__init__()
        self.in_planes = 16

        # 초기 레이어: 3x3 conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Plain 56을 위한 레이어 구성 (n=9)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def plain56():
    """Plain-56: n=9 -> 6*9 + 2 = 56 layers"""
    return PlainNet(PlainBlock, [9, 9, 9])