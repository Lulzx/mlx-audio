# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Encoder for Qwen3 TTS Tokenizer."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderConfig
from .layers import Attention, LayerScale, MLP


class Conv1d(nn.Module):
    """1D Convolution wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        scale = (1.0 / (in_channels * kernel_size)) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale, high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        self.bias = mx.zeros((out_channels,)) if bias else None
        self.stride = stride
        self.padding = padding

    def __call__(self, x: mx.array) -> mx.array:
        x = x.swapaxes(-1, -2)
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])
        y = mx.conv1d(x, self.weight, stride=self.stride)
        if self.bias is not None:
            y = y + self.bias
        return y.swapaxes(-1, -2)


class EncoderLayer0(nn.Module):
    """Initial conv layer."""
    def __init__(self):
        super().__init__()
        self.conv = Conv1d(1, 64, kernel_size=7, padding=3, bias=True)

    def __call__(self, x):
        return self.conv(x)


class EncoderResBlock(nn.Module):
    """Residual block with block.1 and block.3 convs."""
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        # Use a list with indices matching the weight names
        self.block = [
            None,  # block.0 - not used
            Conv1d(channels, hidden, 3, padding=1, bias=True),  # block.1
            None,  # block.2 - not used
            Conv1d(hidden, channels, 1, bias=True),  # block.3
        ]

    def __call__(self, x):
        residual = x
        x = nn.elu(x)
        x = self.block[1](x)
        x = nn.elu(x)
        x = self.block[3](x)
        return x + residual


class EncoderDownsample(nn.Module):
    """Downsample conv layer."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, pad: int = 0):
        super().__init__()
        self.conv = Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=True)

    def __call__(self, x):
        return self.conv(x)


class ConvEncoderLayers(nn.Module):
    """Conv encoder with layers as a list to match weight naming."""
    def __init__(self):
        super().__init__()
        # Layers list - indices must match weight names
        # We need indices 0, 1, 3, 4, 6, 7, 9, 10, 12, 14
        # Use None for gaps
        self.layers = [
            EncoderLayer0(),                           # 0: init conv (1->64)
            EncoderResBlock(64, 32),                   # 1: res block
            None,                                       # 2: gap
            EncoderDownsample(64, 128, 8, 8),          # 3: downsample
            EncoderResBlock(128, 64),                  # 4: res block
            None,                                       # 5: gap
            EncoderDownsample(128, 256, 10, 6, 2),     # 6: downsample
            EncoderResBlock(256, 128),                 # 7: res block
            None,                                       # 8: gap
            EncoderDownsample(256, 512, 12, 5, 3),     # 9: downsample
            EncoderResBlock(512, 256),                 # 10: res block
            None,                                       # 11: gap
            EncoderDownsample(512, 1024, 16, 4, 6),    # 12: downsample
            None,                                       # 13: gap
            EncoderDownsample(1024, 512, 3, 1, 1),     # 14: final conv
        ]

    def __call__(self, x):
        x = self.layers[0](x)   # init conv
        x = self.layers[1](x)   # res block
        x = nn.elu(x)
        x = self.layers[3](x)   # downsample
        x = self.layers[4](x)   # res block
        x = nn.elu(x)
        x = self.layers[6](x)   # downsample
        x = self.layers[7](x)   # res block
        x = nn.elu(x)
        x = self.layers[9](x)   # downsample
        x = self.layers[10](x)  # res block
        x = nn.elu(x)
        x = self.layers[12](x)  # downsample
        x = nn.elu(x)
        x = self.layers[14](x)  # final conv
        return x


class TransformerBlock(nn.Module):
    """Transformer block for encoder."""

    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        intermediate_size: int = 2048,
        head_dim: int = 64,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8000,
        sliding_window: Optional[int] = 250,
        layer_scale: float = 0.01,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=norm_eps)

        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            sliding_window=sliding_window,
        )

        self.mlp = MLP(hidden_size, intermediate_size, act="gelu")
        self.self_attn_layer_scale = LayerScale(hidden_size, layer_scale)
        self.mlp_layer_scale = LayerScale(hidden_size, layer_scale)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.input_layernorm(x)
        x, _ = self.self_attn(x)
        x = self.self_attn_layer_scale(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = self.mlp_layer_scale(x)
        x = residual + x
        return x


class EncoderTransformerLayers(nn.Module):
    """Transformer with layers list."""
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.layers = [
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                head_dim=config.head_dim,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
                sliding_window=config.sliding_window,
                layer_scale=config.layer_scale_initial_scale,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DownsampleConv(nn.Module):
    """Final downsample conv."""
    def __init__(self, dim: int = 512):
        super().__init__()
        self.conv = Conv1d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False)

    def __call__(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """Full encoder matching weight structure."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = ConvEncoderLayers()
        self.encoder_transformer = EncoderTransformerLayers(config)
        self.downsample = DownsampleConv(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.encoder(x)
        x = x.swapaxes(-1, -2)
        x = self.encoder_transformer(x)
        x = x.swapaxes(-1, -2)
        x = self.downsample(x)
        return x
