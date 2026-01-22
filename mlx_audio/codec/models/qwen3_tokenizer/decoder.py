# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Native MLX Decoder for Qwen3 TTS Tokenizer.

Architecture matches the HuggingFace weights exactly:
- Quantizer: RVQ dequantization (16 codebooks)
- Pre-conv: Conv1d (512 -> 1024)
- Pre-transformer: 8 transformer layers
- Upsample: 2x2 = 4x (ConvNeXt-style blocks)
- SEANet decoder: 4 upsample blocks (8x5x4x3 = 480x)
Total: 4 * 480 = 1920x upsampling (12.5 Hz -> 24000 Hz)
"""

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


# =============================================================================
# Quantizer Components
# =============================================================================


class Codebook(nn.Module):
    """Vector quantization codebook with normalized embeddings."""

    def __init__(self, codebook_size: int = 2048, codebook_dim: int = 256):
        super().__init__()
        # Weights stored as embedding_sum / cluster_usage
        self._codebook = CodebookEmbedding(codebook_size, codebook_dim)

    def decode(self, indices: mx.array) -> mx.array:
        """Decode indices to embeddings."""
        return self._codebook.decode(indices)


class CodebookEmbedding(nn.Module):
    """Codebook embedding storage."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.embedding_sum = mx.zeros((codebook_size, codebook_dim))
        self.cluster_usage = mx.ones((codebook_size,))

    @property
    def embeddings(self) -> mx.array:
        """Get normalized embeddings."""
        return self.embedding_sum / (self.cluster_usage[:, None] + 1e-9)

    def decode(self, indices: mx.array) -> mx.array:
        """Decode indices to embeddings."""
        return mx.take(self.embeddings, indices, axis=0)


class VQLayer(nn.Module):
    """Single VQ layer."""

    def __init__(self, codebook_size: int = 2048, codebook_dim: int = 256):
        super().__init__()
        self.layers = [Codebook(codebook_size, codebook_dim)]

    def decode(self, indices: mx.array) -> mx.array:
        """Decode a single codebook."""
        return self.layers[0].decode(indices)


class RVQDecode(nn.Module):
    """RVQ decode module with input/output projections."""

    def __init__(
        self,
        hidden_size: int = 512,
        codebook_dim: int = 256,
        codebook_size: int = 2048,
        num_codebooks: int = 1,
    ):
        super().__init__()
        # 1x1 Conv projections stored as (out, in, 1)
        self.input_proj = Conv1dPointwise(hidden_size, codebook_dim)
        self.output_proj = Conv1dPointwise(codebook_dim, hidden_size)

        # VQ layers
        self.vq = VQMulti(num_codebooks, codebook_size, codebook_dim)

    def __call__(self, codes: mx.array) -> mx.array:
        """Decode codes to latents.

        Args:
            codes: (batch, num_codebooks, frames)

        Returns:
            (batch, frames, hidden_size)
        """
        # Dequantize and sum
        embeddings = self.vq.decode(codes)  # (batch, frames, codebook_dim)
        # Project to hidden_size
        return self.output_proj(embeddings)


class VQMulti(nn.Module):
    """Multiple VQ layers for RVQ."""

    def __init__(
        self, num_codebooks: int, codebook_size: int = 2048, codebook_dim: int = 256
    ):
        super().__init__()
        self.layers = [
            Codebook(codebook_size, codebook_dim) for _ in range(num_codebooks)
        ]

    def decode(self, codes: mx.array) -> mx.array:
        """Decode multiple codebooks and sum.

        Args:
            codes: (batch, num_codebooks, frames)

        Returns:
            (batch, frames, codebook_dim)
        """
        total = None
        for i, codebook in enumerate(self.layers):
            emb = codebook.decode(codes[:, i, :])  # (batch, frames, codebook_dim)
            if total is None:
                total = emb
            else:
                total = total + emb
        return total


class Conv1dPointwise(nn.Module):
    """1x1 convolution (pointwise) matching weight format (out, in, 1)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = mx.zeros((out_channels, in_channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, frames, in_channels)
        weight = self.weight[:, :, 0]  # (out, in)
        return x @ weight.T


class Quantizer(nn.Module):
    """Full quantizer with rvq_first and rvq_rest."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        # First codebook (semantic)
        self.rvq_first = RVQDecode(
            hidden_size=config.hidden_size,
            codebook_dim=config.codebook_dim,
            codebook_size=config.codebook_size,
            num_codebooks=1,
        )
        # Remaining codebooks (acoustic) - 15 codebooks
        self.rvq_rest = RVQDecode(
            hidden_size=config.hidden_size,
            codebook_dim=config.codebook_dim,
            codebook_size=config.codebook_size,
            num_codebooks=config.num_quantizers - 1,
        )

    def __call__(self, codes: mx.array) -> mx.array:
        """Dequantize all codebooks.

        Args:
            codes: (batch, 16, frames)

        Returns:
            (batch, frames, hidden_size)
        """
        first_out = self.rvq_first(codes[:, 0:1, :])
        rest_out = self.rvq_rest(codes[:, 1:, :])
        return first_out + rest_out


# =============================================================================
# Upsample Modules (ConvNeXt-style)
# =============================================================================


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block with depthwise conv."""

    def __init__(self, channels: int):
        super().__init__()
        # Depthwise conv
        self.dwconv = DWConv(channels)
        self.gamma = mx.ones((channels,))
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4, bias=True)
        self.pwconv2 = nn.Linear(channels * 4, channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = x * self.gamma
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        return residual + x


class DWConv(nn.Module):
    """Depthwise 1D conv."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class UpsampleModule(nn.Module):
    """Single upsample stage: ConvTranspose + ConvNeXt block."""

    def __init__(self, channels: int, ratio: int = 2):
        super().__init__()
        # Index 0: ConvTranspose for upsampling
        self.upsample_conv = ConvTranspose1d(
            channels, channels, kernel_size=ratio, stride=ratio
        )
        # Index 1: ConvNeXt block
        self.convnext = ConvNeXtBlock(channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample_conv(x)
        x = self.convnext(x)
        return x


class Upsample(nn.Module):
    """Upsample module with multiple stages."""

    def __init__(self, channels: int, ratios: List[int]):
        super().__init__()
        # List of (ConvTranspose, ConvNeXt) tuples stored as list indices
        self.stages = [UpsampleModule(channels, r) for r in ratios]

    def __call__(self, x: mx.array) -> mx.array:
        for stage in self.stages:
            x = stage(x)
        return x


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


# =============================================================================
# SEANet Decoder (matches HuggingFace weights exactly)
# =============================================================================


class SEANetResidualBlock(nn.Module):
    """SEANet residual block with Snake activation and causal convolution.

    Matches weights: decoder.decoder.{1-4}.block.{2,3,4}.*

    Uses causal (left-only) padding with optional dilation.
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.act1 = Snake(channels)
        self.act2 = Snake(channels)
        # Note: conv1 and conv2 have nested .conv
        # Conv1 uses causal padding with dilation
        self.conv1 = CausalDilatedConv1d(channels, channels, kernel_size, dilation=dilation)
        # Conv2 is kernel_size=1, so no padding needed
        self.conv2 = CausalConv1d(channels, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + residual


class CausalDilatedConv1d(nn.Module):
    """Causal dilated Conv1d.

    Matches PyTorch CausalConvNet with dilation.
    Effective kernel = (kernel_size - 1) * dilation + 1
    Causal padding = effective_kernel - stride
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        # Effective kernel size with dilation
        self.effective_kernel = (kernel_size - 1) * dilation + 1
        # Causal padding: left only
        self.causal_padding = self.effective_kernel - stride
        # Conv with dilation but no padding - we apply causal padding manually
        self.conv = Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - channel-last format
        if self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        return self.conv(x)


class WNConv1d(nn.Module):
    """Weight-normalized Conv1d (stored with nested .conv)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class CausalConv1d(nn.Module):
    """Causal Conv1d with left-only padding (stored with nested .conv).

    Matches PyTorch CausalConvNet: padding = kernel_size - stride (left only).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # Causal padding: left only, no right padding
        self.causal_padding = kernel_size - stride
        # Conv with no padding - we'll apply causal padding manually
        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - channel-last format
        if self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        return self.conv(x)


class SEANetBlock(nn.Module):
    """SEANet upsample block.

    Matches weights: decoder.decoder.{1-4}.block.*
    Structure:
    - block.0: LayerNorm (alpha, beta)
    - block.1: ConvTranspose (in_ch, out_ch, k=stride*2)
    - block.2-4: 3x ResidualBlocks
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # Index 0: Layer norm (stored as alpha/beta not weight/bias)
        self.block = SEANetBlockInner(in_channels, out_channels, stride)

    def __call__(self, x: mx.array) -> mx.array:
        return self.block(x)


class SEANetBlockInner(nn.Module):
    """Inner block structure matching weight indices.

    Structure (block.X):
    - 0: SnakeBeta activation (alpha, beta)
    - 1: ConvTranspose (in_ch -> out_ch)
    - 2, 3, 4: ResidualBlocks with dilations 1, 3, 9
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        # Match weight indices exactly
        # block.0: SnakeBeta activation (NOT LayerNorm!)
        self.snake = BlockSnake(in_channels)
        # block.1: Transposed convolution for upsampling
        self.upsample = WNConv1dTranspose(
            in_channels, out_channels, kernel_size=stride * 2, stride=stride
        )
        # block.2, 3, 4: Residual blocks with dilations 1, 3, 9
        self.residuals = [
            SEANetResidualBlock(out_channels, kernel_size=7, dilation=d)
            for d in [1, 3, 9]
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake(x)
        x = self.upsample(x)
        for res in self.residuals:
            x = res(x)
        return x


class BlockSnake(nn.Module):
    """SnakeBeta activation matching block.0.alpha/beta weights."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


class LayerNormAB(nn.Module):
    """Layer norm with alpha/beta instead of weight/bias."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.alpha = mx.ones((channels,))
        self.beta = mx.zeros((channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        return x * self.alpha + self.beta


class WNConv1dTranspose(nn.Module):
    """Transposed Conv1d with nested .conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class FinalSnake(nn.Module):
    """SnakeBeta activation for final layer.

    Matches weights: decoder.decoder.5.alpha/beta

    SnakeBeta: x + (1/exp(beta)) * sin^2(exp(alpha) * x)
    Note: alpha and beta are stored as log values, so we apply exp().
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - channel-last format
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


class SEANetDecoder(nn.Module):
    """SEANet decoder matching HuggingFace weight structure.

    Structure (decoder.decoder.X):
    - 0: Initial Conv1d (latent_dim -> decoder_dim, k=7)
    - 1-4: 4 upsample blocks with rates [8, 5, 4, 3]
    - 5: Final SnakeBeta activation (alpha, beta)
    - 6: Final Conv1d (96 -> 1, k=7)

    Total upsampling: 8*5*4*3 = 480x
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # Build decoder as list to match weight indices
        self.decoder = []

        # Index 0: Initial conv (latent_dim -> decoder_dim)
        self.decoder.append(InitConv(config.latent_dim, config.decoder_dim))

        # Index 1-4: Upsample blocks
        channels = config.decoder_dim  # 1536
        for i, rate in enumerate(config.upsample_rates):  # [8, 5, 4, 3]
            out_channels = channels // 2
            self.decoder.append(SEANetBlock(channels, out_channels, rate))
            channels = out_channels

        # Index 5: Final SnakeBeta activation (NOT LayerNorm!)
        self.decoder.append(FinalSnake(channels))

        # Index 6: Final conv to audio
        self.decoder.append(FinalConv(channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """Decode latents to audio.

        Args:
            x: (batch, frames, latent_dim)

        Returns:
            (batch, time, 1)
        """
        for module in self.decoder:
            x = module(x)
        return x


class InitConv(nn.Module):
    """Initial conv layer (decoder.decoder.0) with causal padding."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Causal padding for k=7: padding = 7 - 1 = 6 (left only)
        self.causal_padding = 6
        self.conv = Conv1d(in_channels, out_channels, kernel_size=7, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        return self.conv(x)


class FinalConv(nn.Module):
    """Final conv layer (decoder.decoder.6) with causal padding."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Causal padding for k=7: padding = 7 - 1 = 6 (left only)
        self.causal_padding = 6
        self.conv = Conv1d(in_channels, out_channels, kernel_size=7, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
        return self.conv(x)


# =============================================================================
# Pre-Transformer (matches decoder.pre_transformer.*)
# =============================================================================


class PreTransformer(nn.Module):
    """Pre-transformer matching decoder.pre_transformer.* weights."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        # input_proj: (1024 -> 512)
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size, bias=True)

        # 8 transformer layers
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

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # output_proj: (512 -> 1024)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Process through transformer.

        Args:
            x: (batch, frames, latent_dim)

        Returns:
            (batch, frames, latent_dim)
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x, _ = layer(x)

        x = self.norm(x)
        x = self.output_proj(x)
        return x


# =============================================================================
# Full Decoder (matches decoder.* weight structure)
# =============================================================================


class Decoder(nn.Module):
    """Full decoder matching HuggingFace weight structure.

    Structure:
    - quantizer: RVQ dequantization (rvq_first + rvq_rest)
    - pre_conv: Conv1d (hidden_size -> latent_dim)
    - pre_transformer: 8 transformer layers
    - upsample: 2x2 = 4x ConvNeXt-style upsampling
    - decoder: SEANet decoder (480x upsampling)

    Total: 4 * 480 = 1920x (12.5 Hz -> 24000 Hz)
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # Quantizer dequantization
        self.quantizer = Quantizer(config)

        # Pre-conv: (hidden_size -> latent_dim) with CAUSAL padding
        self.pre_conv = CausalConv1d(
            config.hidden_size, config.latent_dim, kernel_size=3, stride=1
        )

        # Pre-transformer
        self.pre_transformer = PreTransformer(config)

        # Upsample (4x total from [2, 2])
        self.upsample = [
            UpsampleStage(config.latent_dim, ratio)
            for ratio in config.upsampling_ratios
        ]

        # SEANet decoder (480x upsampling)
        self.decoder = SEANetDecoder(config)

    def __call__(self, codes: mx.array) -> mx.array:
        """Decode codes to audio waveform.

        Args:
            codes: (batch, num_codebooks, frames)

        Returns:
            (batch, time) audio waveform
        """
        # Dequantize codes to latent
        x = self.quantizer(codes)  # (batch, frames, hidden_size)

        # Pre-conv
        x = self.pre_conv(x)  # (batch, frames, latent_dim)

        # Pre-transformer
        x = self.pre_transformer(x)  # (batch, frames, latent_dim)

        # Upsample 4x
        for up in self.upsample:
            x = up(x)

        # SEANet decode to audio (480x)
        x = self.decoder(x)  # (batch, time, 1)

        # Clamp to valid audio range (matching PyTorch)
        x = mx.clip(x, -1.0, 1.0)

        return x.squeeze(-1)  # (batch, time)


class UpsampleStage(nn.Module):
    """Single upsample stage matching decoder.upsample.X.* weights.

    Structure:
    - 0: ConvTranspose (conv)
    - 1: ConvNeXt block (dwconv, gamma, norm, pwconv1, pwconv2)
    """

    def __init__(self, channels: int, ratio: int = 2):
        super().__init__()
        # Module list with indices 0, 1
        self.conv_transpose = UpsampleConvTranspose(channels, ratio)
        self.convnext = UpsampleConvNeXt(channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_transpose(x)
        x = self.convnext(x)
        return x


class UpsampleConvTranspose(nn.Module):
    """Upsample ConvTranspose matching decoder.upsample.X.0.conv."""

    def __init__(self, channels: int, ratio: int):
        super().__init__()
        self.conv = ConvTranspose1d(channels, channels, kernel_size=ratio, stride=ratio)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class UpsampleConvNeXt(nn.Module):
    """ConvNeXt block matching decoder.upsample.X.1.*."""

    def __init__(self, channels: int):
        super().__init__()
        self.dwconv = DWConv(channels)
        self.gamma = mx.ones((channels,))
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4, bias=True)
        self.pwconv2 = nn.Linear(channels * 4, channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = x * self.gamma
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nn.gelu(x)
        x = self.pwconv2(x)
        return residual + x
