# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    """Configuration for the Qwen3 TTS Tokenizer encoder."""

    audio_channels: int = 1
    sampling_rate: int = 24000
    frame_rate: float = 12.5
    hidden_size: int = 512
    num_filters: int = 64
    codebook_dim: int = 256
    codebook_size: int = 2048
    num_quantizers: int = 32
    num_semantic_quantizers: int = 1
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 2048
    head_dim: int = 64
    hidden_act: str = "gelu"
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 250
    max_position_embeddings: int = 8000
    layer_scale_initial_scale: float = 0.01
    compress: int = 2
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    num_residual_layers: int = 1
    dilation_growth_rate: int = 2
    upsampling_ratios: List[int] = field(default_factory=lambda: [8, 6, 5, 4])
    use_causal_conv: bool = True
    pad_mode: str = "constant"


@dataclass
class DecoderConfig:
    """Configuration for the Qwen3 TTS Tokenizer decoder."""

    latent_dim: int = 1024
    codebook_dim: int = 512
    codebook_size: int = 2048
    decoder_dim: int = 1536
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_quantizers: int = 16
    num_semantic_quantizers: int = 1
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    sliding_window: int = 72
    max_position_embeddings: int = 8000
    layer_scale_initial_scale: float = 0.01
    semantic_codebook_size: int = 4096
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: List[int] = field(default_factory=lambda: [2, 2])


@dataclass
class Qwen3TokenizerConfig:
    """Configuration for the Qwen3 TTS Tokenizer."""

    model_type: str = "qwen3_tts_tokenizer_12hz"
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    frame_rate: float = 12.5
    encode_downsample_rate: int = 1920
    decode_upsample_rate: int = 1920
    encoder_valid_num_quantizers: int = 16
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Qwen3TokenizerConfig":
        """Create config from a dictionary (e.g., from config.json)."""
        encoder_config = EncoderConfig(
            **{
                k: v
                for k, v in config_dict.get("encoder_config", {}).items()
                if k in EncoderConfig.__dataclass_fields__
            }
        )
        decoder_config = DecoderConfig(
            **{
                k: v
                for k, v in config_dict.get("decoder_config", {}).items()
                if k in DecoderConfig.__dataclass_fields__
            }
        )

        return cls(
            model_type=config_dict.get("model_type", "qwen3_tts_tokenizer_12hz"),
            input_sample_rate=config_dict.get("input_sample_rate", 24000),
            output_sample_rate=config_dict.get("output_sample_rate", 24000),
            frame_rate=config_dict.get("encoder_config", {}).get("_frame_rate", 12.5),
            encode_downsample_rate=config_dict.get("encode_downsample_rate", 1920),
            decode_upsample_rate=config_dict.get("decode_upsample_rate", 1920),
            encoder_valid_num_quantizers=config_dict.get(
                "encoder_valid_num_quantizers", 16
            ),
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
