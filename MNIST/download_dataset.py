from torchvision import datasets

mnist = datasets.MNIST(
    root='./data',
    train=True,
    download=True
)