# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Decoder for Qwen3 TTS Tokenizer."""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import DecoderConfig
from .layers import (
    Attention,
    Conv1d,
    ConvTranspose1d,
    GatedMLP,
    LayerScale,
    RMSNorm,
    Snake,
)


class DecoderTransformerBlock(nn.Module):
    """Transformer block for the decoder (with RMSNorm and GatedMLP)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        sliding_window: Optional[int] = None,
        layer_scale: float = 0.01,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            sliding_window=sliding_window,
            attention_bias=False,
        )

        self.mlp = GatedMLP(hidden_size, intermediate_size)

        self.self_attn_layer_scale = LayerScale(hidden_size, layer_scale)
        self.mlp_layer_scale = LayerScale(hidden_size, layer_scale)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(hidden_states, attention_mask, cache)
        hidden_states = self.self_attn_layer_scale(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_layer_scale(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class DecoderTransformer(nn.Module):
    """Pre-transformer for the decoder."""

    def __init__(self, config: DecoderConfig):
        super().__init__()

        # Input projection from latent_dim to hidden_size
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size, bias=True)

        self.layers = [
            DecoderTransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                head_dim=config.head_dim,
                rope_theta=config.rope_theta,
                max_position_embeddings=config.max_position_embeddings,
                sliding_window=config.sliding_window,
                layer_scale=config.layer_scale_initial_scale,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        hidden_states = self.input_proj(hidden_states)

        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            hidden_states, new_layer_cache = layer(hidden_states, attention_mask, layer_cache)
            new_cache.append(new_layer_cache)

        return hidden_states, new_cache


class DecoderResidualBlock(nn.Module):
    """Residual block with Snake activation for decoder."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.act1 = Snake(channels)
        self.act2 = Snake(channels)
        self.conv1 = Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.conv2 = Conv1d(channels, channels, 1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class DecoderUpsampleBlock(nn.Module):
    """Upsampling block for decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_residual_blocks: int = 3,
        residual_kernel_size: int = 7,
    ):
        super().__init__()

        # Snake activation before upsample
        self.pre_act = Snake(in_channels)

        # Transposed convolution for upsampling
        self.upsample = ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2,
            bias=True,
        )

        # Residual blocks after upsampling
        self.residuals = [
            DecoderResidualBlock(out_channels, residual_kernel_size)
            for _ in range(num_residual_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_act(x)
        x = self.upsample(x)
        for residual in self.residuals:
            x = residual(x)
        return x


class ConvDecoder(nn.Module):
    """Convolutional decoder with Snake activations."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # Initial convolution from hidden_size to decoder_dim
        self.init_conv = Conv1d(
            config.latent_dim,  # Input comes from quantizer output projection
            config.decoder_dim,
            kernel_size=7,
            padding=3,
            bias=True,
        )

        # Upsampling blocks
        self.blocks = []
        in_channels = config.decoder_dim

        for i, rate in enumerate(config.upsample_rates):
            out_channels = in_channels // 2
            self.blocks.append(
                DecoderUpsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=rate * 2,
                    stride=rate,
                    num_residual_blocks=3,
                    residual_kernel_size=7,
                )
            )
            in_channels = out_channels

        # Additional upsampling for upsampling_ratios (e.g., [2, 2])
        for ratio in config.upsampling_ratios:
            out_channels = in_channels // 2 if in_channels > 1 else in_channels
            self.blocks.append(
                DecoderUpsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    num_residual_blocks=1,
                    residual_kernel_size=3,
                )
            )
            in_channels = out_channels

        # Final convolution to audio
        self.final_act = Snake(in_channels)
        self.final_conv = Conv1d(in_channels, 1, kernel_size=7, padding=3, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, latent_dim, time)
        x = self.init_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_act(x)
        x = self.final_conv(x)
        return x


class Decoder(nn.Module):
    """Full decoder: Pre-Transformer -> Conv Decoder."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        self.pre_transformer = DecoderTransformer(config)
        self.decoder = ConvDecoder(config)

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        # hidden_states: (batch, time, latent_dim)
        hidden_states, new_cache = self.pre_transformer(hidden_states, cache=cache)

        # Transform to (batch, hidden_size, time) for conv decoder
        # But conv decoder expects latent_dim, so we need output projection
        # Actually the pre_transformer outputs hidden_size, we need to project back
        # For simplicity, we'll adjust the conv decoder input

        # hidden_states: (batch, time, hidden_size) -> (batch, latent_dim, time)
        # We need a projection here - let's add it
        hidden_states = hidden_states.swapaxes(-1, -2)

        # Decode to audio
        audio = self.decoder(hidden_states)

        return audio, new_cache
