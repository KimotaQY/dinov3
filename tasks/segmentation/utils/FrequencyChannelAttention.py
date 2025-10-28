import torch
import torch.nn as nn


class FrequencyChannelAttention(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()

        self.preprocess_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.GELU(),
        )

        self.fca_conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)
        self.fca_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.preprocess_conv(x)

        channel_weights = self.fca_conv(self.fca_pool(features))

        freq_features = torch.fft.fft2(features, norm="backward")
        freq_features = channel_weights * freq_features
        recon_features = torch.fft.ifft2(freq_features,
                                         dim=(-2, -1),
                                         norm="backward")

        output = torch.abs(recon_features)
        return output


if __name__ == "__main__":
    input_tensor = torch.randn(1, 32, 224, 224)
    model = FrequencyChannelAttention(in_channels=32)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
