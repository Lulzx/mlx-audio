# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Native MLX implementation of Qwen3-TTS Talker model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TalkerConfig:
    """Configuration for the Talker model."""

    hidden_size: int = 1024
    text_hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 65536
    vocab_size: int = 3072  # Codec vocab
    text_vocab_size: int = 151936
    num_codebooks: int = 16
    code_predictor_layers: int = 5


class RMSNorm(nn.Module):
    """RMS Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class RotaryEmbedding:
    """Rotary Position Embedding."""

    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 65536):
        self.dim = dim
        self.theta = theta
        self._cos_cache = None
        self._sin_cache = None
        self._cached_seq_len = 0

    def _build_cache(self, seq_len: int):
        if seq_len <= self._cached_seq_len:
            return

        inv_freq = 1.0 / (self.theta ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim))
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, inv_freq)
        self._cos_cache = mx.cos(freqs)
        self._sin_cache = mx.sin(freqs)
        self._cached_seq_len = seq_len

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        seq_len = x.shape[2] + offset
        self._build_cache(seq_len)
        cos = self._cos_cache[offset:seq_len]
        sin = self._sin_cache[offset:seq_len]
        return cos, sin


def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary embeddings to input tensor."""
    # x: (batch, heads, seq_len, head_dim)
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Reshape cos/sin for broadcasting
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim//2)
    sin = sin[None, None, :, :]

    rotated = mx.concatenate([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], axis=-1)
    return rotated


class Attention(nn.Module):
    """Multi-head attention with QK-Norm and GQA."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = RotaryEmbedding(self.head_dim, theta=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Get position offset from cache
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]

        # Apply rotary embeddings
        cos, sin = self.rope(q, offset=offset)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Update cache
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # GQA: expand k, v heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape and project
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)

        return output, new_cache


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_cache


class TextProjection(nn.Module):
    """Projects text embeddings to model hidden size."""

    def __init__(self, text_hidden_size: int, hidden_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(text_hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.linear_fc1(x))
        x = self.linear_fc2(x)
        return x


class CodePredictor(nn.Module):
    """Predicts codebooks 1-15 given the first codebook."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.num_codebooks = config.num_codebooks - 1  # 15 codebooks (1-15)

        # Codec embeddings for each codebook (0-14 for codebooks 1-15)
        self.codec_embedding = [
            nn.Embedding(2048, config.hidden_size)
            for _ in range(self.num_codebooks)
        ]

        # Transformer layers
        self.layers = [
            TransformerBlock(config)
            for _ in range(config.code_predictor_layers)
        ]

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM heads for each codebook
        self.lm_head = [
            nn.Linear(config.hidden_size, 2048, bias=False)
            for _ in range(self.num_codebooks)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        first_codebook: mx.array,
    ) -> mx.array:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) from main model
            first_codebook: (batch, seq_len) first codebook predictions

        Returns:
            codes: (batch, num_codebooks, seq_len) all 15 additional codebooks
        """
        B, L, _ = hidden_states.shape
        all_codes = []

        # Start with hidden states
        x = hidden_states

        for i in range(self.num_codebooks):
            # Add embedding from previous codebook (or first codebook for i=0)
            if i == 0:
                code_emb = self.codec_embedding[i](first_codebook)
            else:
                code_emb = self.codec_embedding[i](all_codes[-1])

            x = x + code_emb

            # Run through transformer layers
            for layer in self.layers:
                x, _ = layer(x, mask=None, cache=None)

            # Normalize and predict
            x_norm = self.norm(x)
            logits = self.lm_head[i](x_norm)

            # Sample or argmax
            codes = mx.argmax(logits, axis=-1)
            all_codes.append(codes)

        # Stack: (batch, 15, seq_len)
        return mx.stack(all_codes, axis=1)


class Talker(nn.Module):
    """Main Talker model for Qwen3-TTS."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Text projection
        self.text_projection = TextProjection(config.text_hidden_size, config.hidden_size)

        # Transformer layers
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Codec head (predicts first codebook)
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Code predictor for remaining codebooks
        self.code_predictor = CodePredictor(config)

    def embed_tokens(self, input_ids: mx.array, is_text: mx.array) -> mx.array:
        """Embed tokens, handling both text and codec tokens."""
        # Text tokens
        text_emb = self.text_embedding(input_ids)
        text_emb = self.text_projection(text_emb)

        # Codec tokens
        codec_emb = self.codec_embedding(input_ids)

        # Select based on token type
        is_text = is_text[:, :, None]  # (batch, seq, 1)
        return mx.where(is_text, text_emb, codec_emb)

    def __call__(
        self,
        input_ids: mx.array,
        is_text: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            is_text: (batch, seq_len) boolean mask for text vs codec tokens
            mask: Optional attention mask
            cache: Optional KV cache

        Returns:
            logits: (batch, seq_len, vocab_size) for first codebook
            new_cache: Updated KV cache
        """
        # Embed tokens
        x = self.embed_tokens(input_ids, is_text)

        # Create causal mask if needed
        if mask is None:
            L = x.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(x.dtype)

        # Run through transformer
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(x, mask=mask, cache=layer_cache)
            new_cache.append(layer_new_cache)

        # Final norm and head
        x = self.norm(x)
        logits = self.codec_head(x)

        return logits, new_cache, x  # Also return hidden states for code predictor


class Qwen3TTSNative(nn.Module):
    """Native MLX Qwen3-TTS model."""

    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config
        self.talker = Talker(config)

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def frame_rate(self) -> float:
        return 12.5

    def generate_codes(
        self,
        input_ids: mx.array,
        is_text: mx.array,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> mx.array:
        """
        Generate codec tokens autoregressively.

        Args:
            input_ids: (batch, seq_len) input token IDs
            is_text: (batch, seq_len) mask for text tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            all_codes: (batch, 16, num_frames) all codebook values
        """
        B = input_ids.shape[0]
        cache = None
        generated_first_codes = []
        hidden_states_list = []

        # Generate first codebook autoregressively
        for _ in range(max_new_tokens):
            logits, cache, hidden = self.talker(
                input_ids, is_text, mask=None, cache=cache
            )

            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Sample
            if top_p < 1.0:
                # Top-p sampling
                sorted_logits = mx.sort(next_logits, axis=-1)[:, ::-1]
                sorted_indices = mx.argsort(next_logits, axis=-1)[:, ::-1]
                cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
                mask = cumulative_probs > top_p
                # Shift mask right to keep at least one token
                mask = mx.concatenate([mx.zeros((B, 1), dtype=mx.bool_), mask[:, :-1]], axis=-1)
                sorted_logits = mx.where(mask, mx.array(-float('inf')), sorted_logits)
                probs = mx.softmax(sorted_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.take_along_axis(sorted_indices, next_token[:, None], axis=-1)[:, 0]
            else:
                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))

            generated_first_codes.append(next_token)
            hidden_states_list.append(hidden[:, -1:, :])

            # Prepare next input
            input_ids = next_token[:, None]
            is_text = mx.zeros_like(input_ids, dtype=mx.bool_)

            # Check for EOS (codec_eos_token_id = 2150)
            if mx.all(next_token == 2150):
                break

        # Stack generated codes
        first_codes = mx.stack(generated_first_codes, axis=1)  # (batch, seq_len)
        hidden_states = mx.concatenate(hidden_states_list, axis=1)  # (batch, seq_len, hidden)

        # Generate remaining codebooks
        remaining_codes = self.talker.code_predictor(hidden_states, first_codes)

        # Combine: first_codes is codebook 0, remaining_codes is codebooks 1-15
        all_codes = mx.concatenate([first_codes[:, None, :], remaining_codes], axis=1)

        return all_codes
