# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Qwen3 TTS Tokenizer - Native MLX Implementation.

This tokenizer encodes audio to discrete codes and decodes codes back to audio.
It uses the official Qwen3-TTS-Tokenizer-12Hz model with 16 codebooks at 12.5 Hz.

The decoder is implemented natively in MLX for fast inference on Apple Silicon.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download

from .config import DecoderConfig, Qwen3TokenizerConfig
from .decoder import Decoder


def _unflatten_dict(d: dict, sep: str = ".") -> dict:
    """Unflatten a dot-separated dictionary into nested dicts."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


class Qwen3TTSTokenizer(nn.Module):
    """Qwen3 TTS Tokenizer for encoding/decoding audio.

    This tokenizer converts audio waveforms to discrete codes and back.
    It operates at 12.5 Hz frame rate with 16 codebooks.

    Native MLX implementation for fast inference on Apple Silicon.

    Attributes:
        sample_rate: Audio sample rate (24000 Hz)
        frame_rate: Codec frame rate (12.5 Hz)
        num_codebooks: Number of codebooks (16)
    """

    def __init__(self, config: Qwen3TokenizerConfig):
        super().__init__()
        self.config = config
        self._decoder = None
        self._weights_loaded = False

    @property
    def sample_rate(self) -> int:
        return self.config.input_sample_rate

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    @property
    def num_codebooks(self) -> int:
        return self.config.encoder_valid_num_quantizers

    def _ensure_decoder(self):
        """Initialize native MLX decoder if not already done."""
        if self._decoder is not None:
            return

        # Use decoder config from tokenizer config
        # It's already a DecoderConfig dataclass
        decoder_cfg = self.config.decoder_config

        # Update num_quantizers to match encoder_valid_num_quantizers
        config = DecoderConfig(
            latent_dim=decoder_cfg.latent_dim,
            codebook_dim=256,  # Actual codebook dim (not the 512 from config)
            codebook_size=decoder_cfg.codebook_size,
            decoder_dim=decoder_cfg.decoder_dim,
            hidden_size=decoder_cfg.hidden_size,
            intermediate_size=decoder_cfg.intermediate_size,
            num_attention_heads=decoder_cfg.num_attention_heads,
            num_key_value_heads=decoder_cfg.num_key_value_heads,
            head_dim=decoder_cfg.head_dim,
            num_hidden_layers=decoder_cfg.num_hidden_layers,
            num_quantizers=self.config.encoder_valid_num_quantizers,
            rms_norm_eps=decoder_cfg.rms_norm_eps,
            rope_theta=decoder_cfg.rope_theta,
            max_position_embeddings=decoder_cfg.max_position_embeddings,
            sliding_window=decoder_cfg.sliding_window,
            layer_scale_initial_scale=decoder_cfg.layer_scale_initial_scale,
            upsample_rates=decoder_cfg.upsample_rates,
            upsampling_ratios=decoder_cfg.upsampling_ratios,
        )

        self._decoder = Decoder(config)

    def _load_weights(self, repo_id: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"):
        """Load weights from HuggingFace."""
        if self._weights_loaded:
            return

        self._ensure_decoder()

        # Download weights
        weights_path = hf_hub_download(repo_id, "model.safetensors")

        # Load and sanitize weights
        weights = mx.load(weights_path)
        decoder_weights = self._sanitize_weights(weights)

        # Load into decoder with strict=False to handle VQ layers
        self._decoder.load_weights(list(decoder_weights.items()), strict=False)
        self._weights_loaded = True

    def _sanitize_weights(self, weights: dict) -> dict:
        """Sanitize weight names for MLX decoder.

        Converts HuggingFace weight names to match our MLX module structure.

        Key mappings:
        - decoder.decoder.X.block.0.alpha -> decoder.decoder.X.block.norm.alpha
        - decoder.decoder.X.block.1.conv -> decoder.decoder.X.block.upsample.conv
        - decoder.decoder.X.block.{2,3,4}.* -> decoder.decoder.X.block.residuals.{0,1,2}.*
        - decoder.upsample.X.0.conv -> upsample.X.conv_transpose.conv
        - decoder.upsample.X.1.* -> upsample.X.convnext.*
        """
        import re

        new_weights = {}

        for key, value in weights.items():
            if not key.startswith("decoder."):
                continue

            # Strip leading "decoder." to get internal path
            new_key = key[8:]  # Remove "decoder."

            # Handle codebook embeddings
            if "_codebook.embedding_sum" in new_key:
                # Get the corresponding cluster_usage
                usage_key = key.replace("embedding_sum", "cluster_usage")
                if usage_key in weights:
                    usage = weights[usage_key]
                    value = value / (usage[:, None] + 1e-9)
                # Keep the key as is for VQ structure
                new_weights[new_key] = value
                continue

            elif "_codebook.cluster_usage" in new_key:
                continue  # Skip, already processed

            # Map upsample module names
            # decoder.upsample.X.0.conv -> upsample.X.conv_transpose.conv
            # decoder.upsample.X.1.dwconv -> upsample.X.convnext.dwconv
            # etc.
            if new_key.startswith("upsample."):
                match = re.match(r"upsample\.(\d+)\.(\d+)\.(.*)", new_key)
                if match:
                    idx, sub_idx, rest = match.groups()
                    if sub_idx == "0":
                        new_key = f"upsample.{idx}.conv_transpose.{rest}"
                    elif sub_idx == "1":
                        new_key = f"upsample.{idx}.convnext.{rest}"

            # Map SEANet decoder block names
            # decoder.X.block.0.alpha -> decoder.decoder.X.block.norm.alpha
            # decoder.X.block.1.conv -> decoder.decoder.X.block.upsample.conv
            # decoder.X.block.{2,3,4}.* -> decoder.decoder.X.block.residuals.{0,1,2}.*
            if new_key.startswith("decoder.") and ".block." in new_key:
                match = re.match(r"decoder\.(\d+)\.block\.(\d+)\.(.*)", new_key)
                if match:
                    block_idx, sub_idx, rest = match.groups()
                    sub_idx = int(sub_idx)
                    if sub_idx == 0:
                        # LayerNorm (alpha/beta)
                        new_key = f"decoder.decoder.{block_idx}.block.norm.{rest}"
                    elif sub_idx == 1:
                        # Upsample conv
                        new_key = f"decoder.decoder.{block_idx}.block.upsample.{rest}"
                    elif sub_idx in [2, 3, 4]:
                        # Residual blocks (2,3,4 -> 0,1,2)
                        res_idx = sub_idx - 2
                        new_key = f"decoder.decoder.{block_idx}.block.residuals.{res_idx}.{rest}"

            # Map simple decoder modules (0, 5, 6)
            # decoder.0.conv -> decoder.decoder.0.conv
            # decoder.5.alpha -> decoder.decoder.5.alpha
            # decoder.6.conv -> decoder.decoder.6.conv
            elif new_key.startswith("decoder."):
                match = re.match(r"decoder\.(\d+)\.(.*)", new_key)
                if match:
                    idx, rest = match.groups()
                    new_key = f"decoder.decoder.{idx}.{rest}"

            # Transpose conv weights to MLX format
            # Regular conv: HF (out, in, k) -> MLX (out, k, in) via (0, 2, 1)
            # Transpose conv: HF (in, out, k) -> MLX (out, k, in) via (1, 2, 0)
            if ".conv.weight" in new_key and value.ndim == 3:
                # ConvTranspose patterns:
                # 1. upsample.X.conv_transpose.conv.weight - pre-SEANet upsample
                # 2. decoder.decoder.X.block.upsample.conv.weight - SEANet block upsample
                is_conv_transpose = (
                    "conv_transpose.conv.weight" in new_key or
                    "block.upsample.conv.weight" in new_key
                )
                if is_conv_transpose:
                    # Transpose conv: (in, out, k) -> (out, k, in)
                    value = mx.transpose(value, (1, 2, 0))
                else:
                    # Regular conv (including dwconv): (out, in, k) -> (out, k, in)
                    value = mx.transpose(value, (0, 2, 1))

            new_weights[new_key] = value

        return new_weights

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio waveform to discrete codes.

        Note: Encoding currently requires PyTorch backend.
        A native MLX encoder is planned for a future release.

        Args:
            audio: Audio waveform of shape (batch, 1, time), (batch, time), or (time,)

        Returns:
            Codes of shape (batch, num_codebooks, frames)
        """
        raise NotImplementedError(
            "Native MLX encoder is not yet implemented. "
            "Encoding requires the PyTorch backend. "
            "For TTS, only the decoder is needed to convert codes to audio."
        )

    def decode(self, codes: mx.array) -> mx.array:
        """Decode discrete codes to audio waveform using native MLX.

        Args:
            codes: Codes of shape (batch, num_codebooks, frames) or (num_codebooks, frames)

        Returns:
            Audio waveform of shape (batch, 1, time)
        """
        self._load_weights()

        # Ensure batch dimension
        if codes.ndim == 2:
            codes = codes[None, :, :]

        # Decode using native MLX
        audio = self._decoder(codes)  # (batch, time)

        # Add channel dimension
        audio = audio[:, None, :]  # (batch, 1, time)

        return audio

    def __call__(self, audio: mx.array) -> Tuple[mx.array, mx.array]:
        """Encode and decode audio (reconstruction).

        Args:
            audio: Input audio waveform

        Returns:
            Tuple of (reconstructed_audio, codes)
        """
        codes = self.encode(audio)
        reconstructed = self.decode(codes)
        return reconstructed, codes

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        config_filename: str = "config.json",
        local_dir: Optional[str] = None,
        **kwargs,
    ) -> "Qwen3TTSTokenizer":
        """Load a pretrained tokenizer.

        Args:
            repo_id: HuggingFace repository ID
            config_filename: Name of the config file
            local_dir: Optional local directory to cache files

        Returns:
            Loaded Qwen3TTSTokenizer instance
        """
        # Download config
        config_file = hf_hub_download(
            repo_id,
            config_filename,
            local_dir=local_dir,
        )

        # Load config
        with open(config_file, "r") as f:
            config_dict = json.load(f)

        config = Qwen3TokenizerConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        return model


# Convenience alias
Tokenizer = Qwen3TTSTokenizer
