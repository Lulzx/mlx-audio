# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Common layers for Qwen3 TTS Tokenizer."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class Conv1d(nn.Module):
    """1D Convolution with proper weight layout for MLX."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # MLX conv weight shape: (out_channels, kernel_size, in_channels // groups)
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - channel-last format (native MLX format)
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])

        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y  # (batch, time, channels) - channel-last


class ConvTranspose1d(nn.Module):
    """1D Transposed Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # MLX conv_transpose weight shape: (out_channels, kernel_size, in_channels // groups)
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - channel-last format (native MLX format)
        y = mx.conv_transpose1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y  # (batch, time, channels) - channel-last


class CausalConv1d(nn.Module):
    """Causal 1D Convolution with left padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.pad_mode = pad_mode

        # Calculate causal padding (left padding only)
        self.causal_padding = (kernel_size - 1) * dilation

        # MLX conv weight shape: (out_channels, kernel_size, in_channels // groups)
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, channels, time) -> (batch, time, channels) for MLX conv
        x = x.swapaxes(-1, -2)

        # Apply causal padding (left only)
        if self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])

        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y.swapaxes(-1, -2)


class Snake(nn.Module):
    """Snake activation function: x + (1/a) * sin^2(a * x)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((channels,))
        self.beta = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - channel-last format
        # alpha/beta shape (channels,) broadcasts with last dim
        return x + (1.0 / (self.beta + 1e-9)) * mx.power(mx.sin(self.alpha * x), 2)


class LayerScale(nn.Module):
    """Per-channel scaling after attention/MLP blocks."""

    def __init__(self, dim: int, init_scale: float = 0.01):
        super().__init__()
        self.scale = mx.ones((dim,)) * init_scale

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        seq_len = x.shape[-2]
        t = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embedding to query and key tensors."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional sliding window."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        sliding_window: Optional[int] = None,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings, rope_theta)
        self.scale = head_dim ** -0.5

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply rotary embeddings
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]
        cos, sin = self.rotary_emb(q, offset=offset)
        q, k = apply_rotary_pos_emb(q, k, cos[None, None, :, :], sin[None, None, :, :])

        # Handle KV cache
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # Repeat KV heads if needed
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Attention
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Apply causal mask
        kv_seq_len = k.shape[2]
        causal_mask = mx.triu(mx.full((seq_len, kv_seq_len), -1e9), k=kv_seq_len - seq_len + 1)
        attn_weights = attn_weights + causal_mask

        # Apply sliding window mask if specified
        if self.sliding_window is not None and kv_seq_len > self.sliding_window:
            window_mask = mx.tril(mx.full((seq_len, kv_seq_len), -1e9), k=-(self.sliding_window + 1))
            attn_weights = attn_weights + window_mask

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output), new_cache


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, act: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = act

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        if self.act == "gelu":
            x = nn.gelu(x)
        elif self.act == "silu":
            x = nn.silu(x)
        else:
            x = nn.gelu(x)
        return self.fc2(x)


class GatedMLP(nn.Module):
    """Gated MLP with SiLU activation (used in decoder)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
