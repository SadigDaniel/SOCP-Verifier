import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


def build_mnist_model(n1: int = 16, n2: int = 32, linear_size: int = 100) -> nn.Sequential:
    """
    Requested structure:

        model = nn.Sequential(
            *conv1,
            *conv2,
            Flatten(),
            nn.Linear(n2*out_width*out_width, linear_size),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    For MNIST, input is 1 x 28 x 28.
    We use two Conv-ReLU blocks with kernel_size=4, stride=2, padding=1.

    Width formula for Conv2d:
        out = floor((in + 2p - k)/s) + 1

    28 -> 14 -> 7, so out_width = 7.
    """
    out_width = 7

    conv1 = [
        nn.Conv2d(1, n1, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
    ]
    conv2 = [
        nn.Conv2d(n1, n2, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
    ]

    model = nn.Sequential(
        *conv1,
        *conv2,
        Flatten(),
        nn.Linear(n2 * out_width * out_width, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10),
    )
    return model




def build_mnist_tiny_model(n1: int = 16, n2: int = 16, linear_size: int = 16) -> nn.Sequential:
    """
    Tiny MNIST MLP for differentiable SOCP / cvxpylayers experiments.

    Architecture:
        Flatten
        Linear(784 -> n1)
        ReLU
        Linear(n1 -> n2)
        ReLU
        Linear(n2 -> 10)

    This has exactly 3 Linear layers, so it fits the original differentiable
    SOCP layer assumption:

        Affine -> ReLU -> Affine -> ReLU -> Affine

    Recommended YAML:
        n1: 16
        n2: 16
        linear_size: 16

    Note:
        linear_size is kept only so the signature matches build_mnist_model.
        It is not used here.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, n1),
        nn.ReLU(),
        nn.Linear(n1, n2),
        nn.ReLU(),
        nn.Linear(n2, 10),
    )
    
# def build_cnn5_model(
#     c1: int = 32,
#     c2: int = 64,
#     c3: int = 128,
#     linear_size: int = 256,
# ) -> nn.Sequential:
#     """
#     CNN-5 architecture for CIFAR-10 (3 x 32 x 32 input).

#     Model architecture:

#         Input
#           │
#           ├── Conv3x3(3  → c1)
#           ├── ReLU
#           ├── Conv3x3(c1 → c1)
#           ├── ReLU
#           ├── MaxPool2d(2)
#           │
#           ├── Conv3x3(c1 → c2)
#           ├── ReLU
#           ├── Conv3x3(c2 → c2)
#           ├── ReLU
#           ├── MaxPool2d(2)
#           │
#           ├── Conv3x3(c2 → c3)
#           ├── ReLU
#           ├── AdaptiveAvgPool2d(1)
#           │
#           ├── Flatten
#           ├── Linear(c3 → linear_size)
#           ├── ReLU
#           └── Linear(linear_size → 10)

#     Feature map sizes:

#         3x32x32
#             ↓
#         c1 x32x32
#             ↓
#         c1 x32x32
#             ↓
#         c1 x16x16
#             ↓
#         c2 x16x16
#             ↓
#         c2 x16x16
#             ↓
#         c2 x8x8
#             ↓
#         c3 x8x8
#             ↓
#         c3 x1x1
#             ↓
#         Linear -> 10 classes
#     """

#     return nn.Sequential(
#         # Block 1
#         nn.Conv2d(3, c1, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(c1, c1, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),

#         # Block 2
#         nn.Conv2d(c1, c2, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Conv2d(c2, c2, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),

#         # Block 3
#         nn.Conv2d(c2, c3, kernel_size=3, padding=1),
#         nn.ReLU(),

#         # Global pooling
#         nn.AdaptiveAvgPool2d((1, 1)),

#         Flatten(),

#         # Classifier
#         nn.Linear(c3, linear_size),
#         nn.ReLU(),
#         nn.Linear(linear_size, 10),
#     )

def build_cnn5_model(
    c1: int = 16,
    c2: int = 32,
    c3: int = 64,
    linear_size: int = 128,
    num_classes: int = 10,
):
    """
    Small pooling-free CNN-5 for CIFAR-10 certified training.

    Input:
        3 x 32 x 32

    Spatial sizes:
        32 -> 16 -> 16 -> 8 -> 8 -> 4

    The architecture only uses Conv2d, ReLU, Flatten, and Linear so it
    remains compatible with the current dual-WK and SOCP implementations.
    """
    return nn.Sequential(
        nn.Conv2d(
            3,
            c1,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.ReLU(),

        nn.Conv2d(
            c1,
            c1,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(),

        nn.Conv2d(
            c1,
            c2,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.ReLU(),

        nn.Conv2d(
            c2,
            c2,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(),

        nn.Conv2d(
            c2,
            c3,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.ReLU(),

        nn.Flatten(),

        nn.Linear(
            c3 * 4 * 4,
            linear_size,
        ),
        nn.ReLU(),

        nn.Linear(
            linear_size,
            num_classes,
        ),
    )
    

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes=32, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride, padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=1,
            bias=True,
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes,
                    kernel_size=1, stride=stride,
                    bias=True,
                )
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + self.shortcut(x))


class ResNet4b(nn.Module):
    """CIFAR-10 ResNet-4b with four residual blocks and no batch normalization."""

    def __init__(self):
        super().__init__()

        # Input: 3 x 32 x 32 -> 16 x 16 x 16
        self.conv1 = nn.Conv2d(
            3, 16,
            kernel_size=3, stride=2, padding=1,
            bias=True,
        )

        # 16 x 16 x 16 -> 32 x 8 x 8
        self.layer1 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1),
        )

        # 32 x 8 x 8 -> 32 x 4 x 4
        self.layer2 = nn.Sequential(
            BasicBlock(32, 32, stride=2),
            BasicBlock(32, 32, stride=1),
        )

        self.linear1 = nn.Linear(32 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


def build_resnet4b_model():
    return ResNet4b()