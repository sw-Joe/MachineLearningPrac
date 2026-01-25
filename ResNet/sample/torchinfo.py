import torch
from torchinfo import summary

from ResNet.building_block import BasicBlock, ResNet




# Factory functions for different ResNet variants
def resnet18(num_classes=1000):
    """ResNet-18: 2+2+2+2 = 8 residual blocks + initial layers"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
    """ResNet-34: 3+4+6+3 = 16 residual blocks + initial layers"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


# Example usage
if __name__ == "__main__":
    # Create a ResNet-18 model
    model = resnet18(num_classes=1000)  # For CIFAR-10
    summary(model)

    
    # Test with a random input
    x = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [1, 10]