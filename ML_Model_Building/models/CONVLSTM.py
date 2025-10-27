
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        cell_next = f * c + i * g
        h_next = o * torch.tanh(cell_next)

        return h_next, cell_next


class ConvLSTM(nn.Module):
    def __init__(self, input_channel=1, hidden_channels=16, kernel_size=3, seq_len=7):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.seq_len = seq_len
        self.cell = ConvLSTMCell(input_channel, hidden_channels, kernel_size)
        self.conv_out = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, C, H, W = x.size()
        h = torch.zeros(batch_size, self.hidden_channels,
                        H, W, device=x.device)
        c = torch.zeros(batch_size, self.hidden_channels,
                        H, W, device=x.device)

        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)

        out = self.conv_out(h)
        return out


class ConvLSTM2Layers(nn.Module):
    def __init__(self, input_channel=1, hidden_channels=16, kernel_size=3, seq_len=7):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.seq_len = seq_len
        self.cell1 = ConvLSTMCell(input_channel, hidden_channels, kernel_size)
        self.cell2 = ConvLSTMCell(
            hidden_channels, hidden_channels*2, kernel_size)
        self.conv_out = nn.Conv2d(hidden_channels*2, 1, 1)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, C, H, W = x.size()
        h1 = torch.zeros(batch_size, self.hidden_channels,
                         H, W, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_channels,
                         H, W, device=x.device)

        h2 = torch.zeros(batch_size, self.hidden_channels*2,
                         H, W, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_channels*2,
                         H, W, device=x.device)
        for t in range(seq_len):
            h1, c1 = self.cell1(x[:, t], h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
        out = self.conv_out(h2)
        return out
