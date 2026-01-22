# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Qwen3-TTS Model for text-to-speech synthesis."""

from .config import ModelConfig
from .qwen3_tts import Model

__all__ = ["Model", "ModelConfig"]
