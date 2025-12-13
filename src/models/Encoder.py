import torch
import torch.nn as nn


class SpatioAttention(nn.Module):
    """
    Creates a Spatial Attention of the feature maps, implements simple attention tho (not the transformers one) will be implemented next

    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(x)
        attn = self.sigmoid(attn)

        return x * attn


class GridEncoder_SpatioAttention(nn.Module):
    """
    Encoder that will extract data from the given input from dataset, enhance the extracted data with attention layers

    """

    def __init__(self, input_channels=1, hidden_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels*2)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels*2, kernel_size=3, padding=1)
        self.attn = SpatioAttention(hidden_channels*2)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attn(x)
        x = self.pool(x)

        return x
