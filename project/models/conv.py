from pathlib import PureWindowsPath
import torch
import torch.nn as nn
from typing import List, Union
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, channels: int, n_classes: int, dim_sizes: List[int],
                 kernel_size: int, stride: int, padding: int, **kwargs):
        super(BaseModel, self).__init__()
        conv_stack = []

        self.dim_sizes = dim_sizes
        self.IMG_SIZE = 32
        d_s = [channels] + dim_sizes  # Add Channels to the list of layer sizes
        for i, d_in in enumerate(d_s[:-1]):
            d_out = d_s[i+1]
            layer = ConvSingle(d_in, d_out, kernel_size, stride, padding)
            conv_stack.append(layer)

        self.conv_stack = nn.Sequential(*conv_stack)
        self.f_h, self.f_w = self.check_final_size()

        self.fc = nn.Linear(self.dim_sizes[-1]*self.f_h * self.f_w, n_classes)

    def forward(self, x):
        out = self.conv_stack(x)
        out = out.view(-1, self.dim_sizes[-1]*self.f_h * self.f_w)
        return self.fc(out)

    def check_final_size(self):

        sample = torch.randn(1, 3, 32, 32)
        try:
            out = self.conv_stack(sample)
        except ValueError:
            raise ValueError("Insufficient H and W in output, reduce number of layers "
                             "or reduce stride.")
        return out.size()[2:]


class StackModel(BaseModel):
    """StackModel convolutional network using Convolutional Stacks instead of 
    single layers. Model inherits from BaseModel instead of nn.Module since the 
    forward and check_final_size methods are the same

    Args:
        BaseModel (nn.Module): Base Convolutional Model
    """

    def __init__(self, channels, n_classes, dim_sizes, kernel_size, stride, padding, **kwargs):
        super(BaseModel, self).__init__()
        conv_stack = []

        self.dim_sizes = dim_sizes
        self.IMG_SIZE = 32
        d_s = [channels] + dim_sizes  # Add Channels to the list of layer sizes
        for i, d_in in enumerate(d_s[:-1]):
            d_out = d_s[i+1]
            layer = ConvPoolStack(d_in, d_out, kernel_size, stride, padding)
            conv_stack.append(layer)

        self.conv_stack = nn.Sequential(*conv_stack)
        self.f_h, self.f_w = self.check_final_size()

        self.fc = nn.Linear(self.dim_sizes[-1]*self.f_h * self.f_w, n_classes)


class ConvSingle(nn.Module):
    def __init__(self, d_in, d_out, kernel_size, stride, padding):
        super(ConvSingle, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_in, d_out,
                      kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(d_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.net(x)


class ConvPoolStack(nn.Module):
    """
    Convolutional Stack with Pooling and Residual Connection
    """    # Convolutional Stack that results

    def __init__(self, d_in, d_out, kernel_size, stride, padding):
        super(ConvPoolStack, self).__init__()

        self.conv1 = nn.Conv2d(d_in, d_in,
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(d_in, d_in,
                               kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(d_in, d_out, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(stride)
        self.norm = nn.BatchNorm2d(d_in)
        self.norm2 = nn.BatchNorm2d(d_out)

    def forward(self, x):

        out1 = F.gelu(self.norm(self.conv1(x)))

        out2 = F.gelu(self.norm(self.conv2(out1)))
        out3 = F.gelu(self.norm2(self.conv3(x+out2)))
        final = self.pool(out3)
        return final
