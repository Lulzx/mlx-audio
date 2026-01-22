# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Configuration for Qwen3-TTS models."""

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for Qwen3-TTS model.

    This model has a complex nested config with talker_config containing
    the actual transformer architecture parameters.
    """

    # Model type identification
    model_type: str = "qwen3_tts_talker"
    tts_model_type: str = "custom_voice"  # "base", "custom_voice", or "voice_design"
    tts_model_size: str = "1b7"  # "0b6" or "1b7"

    # Model architecture (from talker_config)
    hidden_size: int = 1024
    text_hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # Vocabulary
    vocab_size: int = 3072  # Audio codec vocab
    text_vocab_size: int = 151936  # Text tokenizer vocab

    # Position embeddings
    max_position_embeddings: int = 65536
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict] = None

    # Codec settings
    num_code_groups: int = 16  # Number of codebooks
    position_id_per_seconds: int = 13  # ~12.5 Hz frame rate

    # Special tokens
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    tts_pad_token_id: int = 151671
    codec_bos_id: int = 2149
    codec_eos_token_id: int = 2150
    codec_pad_id: int = 2148
    codec_think_id: int = 2154
    codec_nothink_id: int = 2155
    codec_think_bos_id: int = 2156
    codec_think_eos_id: int = 2157

    # Language IDs
    codec_language_id: Dict[str, int] = field(
        default_factory=lambda: {
            "chinese": 2055,
            "english": 2050,
            "japanese": 2058,
            "korean": 2064,
            "german": 2053,
            "french": 2061,
            "russian": 2069,
            "portuguese": 2071,
            "spanish": 2054,
            "italian": 2070,
            "beijing_dialect": 2074,
            "sichuan_dialect": 2062,
        }
    )

    # Speaker IDs (for CustomVoice model)
    spk_id: Dict[str, int] = field(
        default_factory=lambda: {
            "vivian": 3065,
            "serena": 3066,
            "uncle_fu": 3010,
            "dylan": 2878,
            "eric": 2875,
            "ryan": 3061,
            "aiden": 2861,
            "ono_anna": 2873,
            "sohee": 2864,
        }
    )

    # Speaker dialect info
    spk_is_dialect: Dict[str, str] = field(
        default_factory=lambda: {
            "vivian": False,
            "serena": False,
            "uncle_fu": False,
            "dylan": "beijing_dialect",
            "eric": "sichuan_dialect",
            "ryan": False,
            "aiden": False,
            "ono_anna": False,
            "sohee": False,
        }
    )

    # Audio settings
    sample_rate: int = 24000

    # Generation settings
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = False

    @property
    def supported_speakers(self) -> List[str]:
        """List of supported speaker names."""
        return list(self.spk_id.keys())

    @property
    def supported_languages(self) -> List[str]:
        """List of supported languages."""
        return list(self.codec_language_id.keys())

    @classmethod
    def from_dict(cls, params: dict) -> "ModelConfig":
        """Create config from a dictionary.

        Handles the nested talker_config structure of Qwen3-TTS models.
        """
        # Extract talker_config if present (nested config structure)
        if "talker_config" in params:
            talker = params["talker_config"]
            # Flatten nested config - talker params take precedence
            flat_params = {**params}
            flat_params.update(talker)
            del flat_params["talker_config"]
            # Remove code_predictor_config as we don't use it directly
            if "code_predictor_config" in flat_params:
                del flat_params["code_predictor_config"]
            params = flat_params

        # Filter to only valid fields
        valid_fields = set(inspect.signature(cls).parameters.keys())
        filtered = {k: v for k, v in params.items() if k in valid_fields}

        return cls(**filtered)
