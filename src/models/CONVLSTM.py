import torch
import torch.nn as nn


class ConvLSTM_NLAYERS(nn.Module):
    def __init__(self, input_channel=1, hidden_channels=[16, 32, 64], kernel_size=3):
        """
        hidden_channels: list → number of ConvLSTM layers = len(hidden_channels)
        e.g. [16, 32, 64] → 3 ConvLSTM layers
        """
        super().__init__()

        self.num_layers = len(hidden_channels)
        self.hidden_channels = hidden_channels
        layers = []
        in_channels = input_channel

        for h in hidden_channels:
            layers.append(ConvLSTMCell(in_channels, h, kernel_size))
            in_channels = h  # next layer receives previous layer's hidden states

        self.layers = nn.ModuleList(layers)

        # Output layer converts last hidden state to 1-channel prediction
        self.conv_out = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        """
        x: (batch, seq_len, C, H, W)
        """
        batch_size, seq_len, _, H, W = x.shape

        # initialize hidden states for all layers
        h = []
        c = []
        for h_ch in self.hidden_channels:
            h.append(torch.zeros(batch_size, h_ch, H, W, device=x.device))
            c.append(torch.zeros(batch_size, h_ch, H, W, device=x.device))

        # process sequence
        for t in range(seq_len):
            inp = x[:, t]

            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]    # feed output to next layer

        # final output from last layer
        out = self.conv_out(h[-1])
        return out, h[-1]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels, kernel_size, padding=padding, bias=True)

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
    def __init__(self, input_channel=1, hidden_channels=[32, 64], kernel_size=3, seq_len=7):
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
    def __init__(self, input_channel=1, hidden_channels=20, kernel_size=3, seq_len=7):
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
