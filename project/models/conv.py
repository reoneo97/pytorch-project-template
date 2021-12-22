import torch.nn as nn 
from typing import List, Union
import torch.nn.functional as F


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
    # Convolutional Stack that results
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
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm2d(d_in)
        self.norm2 = nn.BatchNorm2d(d_out)

    def forward(self, x):

        out1 = F.gelu(self.norm(self.conv1(x)))

        out2 = F.gelu(self.norm(self.conv2(out1)))
        out3 = F.gelu(self.norm2(self.conv3(x+out2)))
        final = self.pool(out3)
        # logger.info(final.shape)
        # logger.info(x.shape)
        return final

