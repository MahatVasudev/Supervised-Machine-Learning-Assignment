# INFO: This model is going to be used for encoding the data, to a smaller bit
# manageable map, so that it does not eat much processing and space

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import SpatioAttention


class AutoEncoderFire(nn.Module):
    def __init__(
        self, in_channel: int, bottleneck_channel: int, hidden_channels: list[int]
    ) -> None:
        """
        #################################################
        #################################################

        AutoEncoderFire: Auto Encoder Model Where we have We determine the perfect bottleneck which describes out model

        #############
        # Architecture:

        in_channels -> [hidden_channel 1, hidden_channel_2, ..., hidden_channel_n] -> bottleneck -> [hidden_channel_n, hidden_channel_n-1, ..., hidden_channel_1] -> out_channel (same as in_channel no.)

        #############

        # Parameters:
        in_channel: int -> in_channels, same (out_channels)
        bottleneck_channel: int -> size of bottleneck (encoded)
        hidden_channels: list(int...) -> list of hidden channel sizes (reversed for decoder)

        # Methods:

        train, predict: takes raw data returns decoded data
        decode: takes in bottleneck data returns decoded data
        encode: takes in raw data, returns bottlneck (encoded) data
        """

        # Stupid Thing
        super().__init__()

        # Predefined parameters
        self.max_channels = 20

        # Assertions
        assert (
            len(hidden_channels) <= self.max_channels
        ), f"hidden channels can not be no more than MAX_CHANNELS {
            self.max_channels}"

        self.in_channels = in_channel
        self.bottleneck_channels = bottleneck_channel
        self.hidden_channels = hidden_channels

        self.__hidden_channels_gen()

        self.bottleneck_encoder = AutoEncoderLayer(
            in_channel=self.hidden_channels[-1],
            out_channel=self.bottleneck_channels,
            use_attention=True,
            is_encoder=True,
        )

    def __hidden_channels_gen(self) -> None:

        encoder_listed = []
        decoder_listed = []
        in_chan = self.in_channels
        for hidden_channel_size in self.hidden_channels:
            encoder_listed.append(
                AutoEncoderLayer(
                    in_channel=in_chan,
                    out_channel=hidden_channel_size,
                    use_attention=True,
                    is_encoder=True,
                )
            )
            in_chan = hidden_channel_size

        in_chan = self.bottleneck_channels
        for hidden_channel_size_dec in reversed(self.hidden_channels):
            decoder_listed.append(
                AutoEncoderLayer(
                    in_channel=in_chan,
                    out_channel=hidden_channel_size_dec,
                    use_attention=True,
                    is_encoder=False,
                )
            )
            in_chan = hidden_channel_size_dec

        decoder_listed.append(
            AutoEncoderLayer(
                in_channel=in_chan,
                out_channel=self.in_channels,
                use_attention=False,
                is_encoder=False,
            )
        )
        self.encoder_channels = nn.ModuleList(encoder_listed)
        self.decoder_channels = nn.ModuleList(decoder_listed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded_x = self.decode(self.encode(x))
        return decoded_x[:, :, : x.shape[-2], : x.shape[-1]]

    def decode(self, bottleneck: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_channels:
            bottleneck = layer(bottleneck)

        return bottleneck

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder_channels:
            x = layer(x)
        return self.bottleneck_encoder(x)


class AutoEncoderLayer(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, use_attention: bool, is_encoder: bool
    ):
        """
        ################################################
        ################################################

        AutoEncoderLayer: fundamental layer for encoder and decoder layers, Uses attention layer check models.Encoder.SpatioAttention
        Use of Attention Layers are Optional
        ###############

        --- WARN: uses batch normalization, suggested to use IN and OUT channels as multiples of 8 ---
        UPDATE: remove batch normalization and instead moved to groupnorm, because of small batch size
        """

        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_attention = use_attention
        if is_encoder:
            self.conv = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
                stride=2,
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=1,
            )

        # self.bn = nn.BatchNorm2d(out_channel)
        num_groups = min(8, out_channel)
        self.gn = nn.GroupNorm(num_groups, out_channel)

        if self.use_attention:
            self.attn = SpatioAttention(out_channel)
        else:
            self.attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = F.relu(self.bn(self.conv(x)))

        x = F.relu(self.gn(self.conv(x)))

        if self.use_attention and self.attn is not None:
            x = self.attn(x)

        return x


if __name__ == "__main__":
    X = torch.ones((1, 1, 685, 256), dtype=torch.float32)
    model = AutoEncoderFire(
        in_channel=1, bottleneck_channel=128, hidden_channels=[32, 64, 128]
    )

    print(model(X).shape)
