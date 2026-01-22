# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Qwen3 TTS Tokenizer.

This tokenizer encodes audio to discrete codes and decodes codes back to audio.
It uses the official Qwen3-TTS-Tokenizer-12Hz model with 16 codebooks at 12.5 Hz.

Note: Currently uses PyTorch/transformers backend for the neural network operations.
A full native MLX implementation is planned for a future release.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download

from .config import Qwen3TokenizerConfig


class Qwen3TTSTokenizer(nn.Module):
    """Qwen3 TTS Tokenizer for encoding/decoding audio.

    This tokenizer converts audio waveforms to discrete codes and back.
    It operates at 12.5 Hz frame rate with 16 codebooks.

    Currently uses the transformers library for model operations.
    Requires: pip install transformers torch

    Attributes:
        sample_rate: Audio sample rate (24000 Hz)
        frame_rate: Codec frame rate (12.5 Hz)
        num_codebooks: Number of codebooks (16)
    """

    def __init__(self, config: Qwen3TokenizerConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._processor = None

    @property
    def sample_rate(self) -> int:
        return self.config.input_sample_rate

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    @property
    def num_codebooks(self) -> int:
        return self.config.encoder_valid_num_quantizers

    def _load_model(self):
        """Load the model using transformers library."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._model = AutoModel.from_pretrained(
                "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            self._model.eval()

        except ImportError as e:
            raise ImportError(
                "Qwen3TTSTokenizer requires PyTorch and transformers. "
                "Install with: pip install torch transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen3-TTS-Tokenizer-12Hz: {e}. "
                "Make sure you have internet access and the model is available."
            ) from e

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio waveform to discrete codes.

        Args:
            audio: Audio waveform of shape (batch, 1, time), (batch, time), or (time,)

        Returns:
            Codes of shape (batch, num_codebooks, frames)
        """
        import numpy as np
        import torch

        self._load_model()

        # Prepare audio
        if audio.ndim == 1:
            audio_np = np.array(audio)[None, :]
        elif audio.ndim == 2:
            audio_np = np.array(audio)
        elif audio.ndim == 3:
            audio_np = np.array(audio).squeeze(1)
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_np).float()

        # Encode
        with torch.no_grad():
            outputs = self._model.encode(audio_tensor)

        # Get codes - structure depends on model output
        if hasattr(outputs, 'audio_codes'):
            codes = outputs.audio_codes
        elif isinstance(outputs, tuple):
            codes = outputs[0]
        else:
            codes = outputs

        # Convert to numpy then MLX
        if hasattr(codes, 'cpu'):
            codes_np = codes.cpu().numpy()
        else:
            codes_np = np.array(codes)

        # Ensure shape is (batch, codebooks, frames)
        if codes_np.ndim == 2:
            codes_np = codes_np[None, :, :]

        return mx.array(codes_np)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode discrete codes to audio waveform.

        Args:
            codes: Codes of shape (batch, num_codebooks, frames) or (num_codebooks, frames)

        Returns:
            Audio waveform of shape (batch, 1, time)
        """
        import numpy as np
        import torch

        self._load_model()

        # Prepare codes
        if codes.ndim == 2:
            codes_np = np.array(codes)[None, :, :]
        else:
            codes_np = np.array(codes)

        # Convert to torch
        codes_tensor = torch.from_numpy(codes_np).long()

        # Decode
        with torch.no_grad():
            audio = self._model.decode(codes_tensor)

        # Convert to numpy then MLX
        if hasattr(audio, 'cpu'):
            audio_np = audio.cpu().numpy()
        elif isinstance(audio, tuple):
            audio_np = audio[0].cpu().numpy() if hasattr(audio[0], 'cpu') else np.array(audio[0])
        else:
            audio_np = np.array(audio)

        # Ensure shape is (batch, 1, time)
        if audio_np.ndim == 1:
            audio_np = audio_np[None, None, :]
        elif audio_np.ndim == 2:
            audio_np = audio_np[:, None, :]

        return mx.array(audio_np)

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
