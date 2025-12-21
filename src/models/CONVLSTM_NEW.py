import torch
import torch.nn as nn
from .CONVLSTM import ConvLSTM_NLAYERS
from .Auto_Encoder import AutoEncoderFire


class CONVLSTM_FIREMODEL(nn.Module):
    def __init__(self, auto_encoder: AutoEncoderFire, hidden_channels=256):
        super().__init__()

        self.auto_encoder = auto_encoder

        self.convlstm = ConvLSTM_NLAYERS(
            input_channel=hidden_channels,
            hidden_channels=[
                hidden_channels//4,
                hidden_channels//2,
                hidden_channels
            ],
            kernel_size=3,
        )
        self.latent_proj = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        encoded_frames = []

        for t in range(T):
            f = self.auto_encoder.encode(x[:, t])
            encoded_frames.append(f)

        encoded_seq = torch.stack(encoded_frames, dim=1)

        pred, features = self.convlstm(encoded_seq)

        features = self.latent_proj(features)

        out = self.auto_encoder.decode(features)[:, :, :H, :W]

        return out
