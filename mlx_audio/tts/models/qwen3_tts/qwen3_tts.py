# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Qwen3-TTS Model implementation for MLX.

Supports both native MLX inference (fast) and PyTorch backend (fallback).
"""

import time
from pathlib import Path
from typing import Generator, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import GenerationResult
from .config import ModelConfig
from .model import Qwen3TTSNative, TalkerConfig


class Model(nn.Module):
    """Qwen3-TTS Model for text-to-speech synthesis.

    Native MLX implementation for fast inference on Apple Silicon.

    Supports three model variants:
    - Base: Voice cloning with reference audio
    - CustomVoice: Preset speakers with instruction-based emotion control
    - VoiceDesign: Natural language voice description
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # Build native model config
        talker_config = TalkerConfig(
            hidden_size=config.hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            vocab_size=config.vocab_size,
            text_vocab_size=config.text_vocab_size,
            num_codebooks=config.num_code_groups,
        )

        # Native MLX model
        self.talker = Qwen3TTSNative(talker_config).talker

        # Tokenizer (loaded in post_load_hook)
        self.tokenizer = None
        self._audio_tokenizer = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.config.position_id_per_seconds

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Load tokenizer after weights are loaded."""
        from transformers import AutoTokenizer

        model.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        return model

    def sanitize(self, weights: dict) -> dict:
        """Remap weight names from HuggingFace format to MLX format."""
        new_weights = {}

        for k, v in weights.items():
            # All weights start with "talker."
            if not k.startswith("talker."):
                continue

            new_key = k

            # Handle list-based modules (code_predictor embeddings and lm_heads)
            # talker.code_predictor.model.codec_embedding.0.weight -> talker.code_predictor.codec_embedding.0.weight
            new_key = new_key.replace("code_predictor.model.", "code_predictor.")

            # talker.model.layers -> talker.layers
            new_key = new_key.replace("talker.model.", "talker.")

            new_weights[new_key] = v

        return new_weights

    @property
    def audio_tokenizer(self):
        """Lazily load the audio tokenizer for decoding."""
        if self._audio_tokenizer is None:
            from mlx_audio.codec.models.qwen3_tokenizer import Qwen3TTSTokenizer

            self._audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                "Qwen/Qwen3-TTS-Tokenizer-12Hz"
            )
        return self._audio_tokenizer

    def get_speaker_token(self, speaker: str) -> int:
        """Get speaker token ID."""
        spk_lower = speaker.lower().replace(" ", "_")
        return self.config.spk_id.get(spk_lower)

    def get_language_token(self, language: str) -> int:
        """Get language token ID."""
        lang_lower = language.lower().replace(" ", "_")
        return self.config.codec_language_id.get(
            lang_lower, self.config.codec_language_id["english"]
        )

    def prepare_prompt(
        self,
        text: str,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        language: str = "english",
    ) -> tuple:
        """Prepare input tokens for generation.

        Returns:
            input_ids: Token IDs
            is_text: Boolean mask (True for text tokens, False for codec)
        """
        tokens = []
        is_text_mask = []

        # TTS BOS
        tokens.append(self.config.tts_bos_token_id)
        is_text_mask.append(True)

        # Language token
        lang_id = self.get_language_token(language)
        tokens.append(lang_id)
        is_text_mask.append(False)

        # Speaker token (if CustomVoice)
        if speaker:
            spk_id = self.get_speaker_token(speaker)
            if spk_id:
                tokens.append(spk_id)
                is_text_mask.append(False)

        # Instruction (if provided)
        if instruct:
            instruct_ids = self.tokenizer.encode(instruct, add_special_tokens=False)
            tokens.extend(instruct_ids)
            is_text_mask.extend([True] * len(instruct_ids))

        # Main text
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(text_ids)
        is_text_mask.extend([True] * len(text_ids))

        # Codec BOS
        tokens.append(self.config.codec_bos_id)
        is_text_mask.append(False)

        input_ids = mx.array([tokens])
        is_text = mx.array([is_text_mask])

        return input_ids, is_text

    def generate_result(
        self,
        audio: mx.array,
        start_time: float,
        token_count: int,
        segment_idx: int,
    ) -> GenerationResult:
        """Create a GenerationResult from generated audio."""
        samples = audio.shape[-1] if audio is not None and audio.ndim > 0 else 0
        if samples == 0:
            raise ValueError("No audio generated")

        sample_rate = self.sample_rate
        audio_duration_seconds = samples / sample_rate
        elapsed_time = time.perf_counter() - start_time
        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        duration_hours = int(audio_duration_seconds // 3600)
        duration_mins = int((audio_duration_seconds % 3600) // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        return GenerationResult(
            audio=audio.squeeze() if audio.ndim > 1 else audio,
            samples=samples,
            sample_rate=sample_rate,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed_time, 2) if elapsed_time > 0 else 0,
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        language: str = "english",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2000,
        split_pattern: str = "\n",
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text using native MLX.

        Args:
            text: Text to synthesize
            voice: Speaker name for CustomVoice model
            ref_audio: Reference audio for voice cloning (Base model)
            ref_text: Transcript of reference audio
            instruct: Emotion/style instruction
            language: Target language
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum codec tokens to generate
            split_pattern: Pattern to split text
            verbose: Show progress

        Yields:
            GenerationResult with audio and metadata
        """
        # Split text into segments
        prompt_text = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = [p for p in prompt_text.split(split_pattern) if p.strip()]

        for segment_idx, segment_text in enumerate(prompts):
            time_start = time.perf_counter()

            if verbose:
                print(f"Generating segment {segment_idx + 1}/{len(prompts)}: {segment_text[:50]}...")

            # Prepare input
            input_ids, is_text = self.prepare_prompt(
                text=segment_text,
                speaker=voice,
                instruct=instruct,
                language=language,
            )

            # Generate codes
            codes = self._generate_codes(
                input_ids=input_ids,
                is_text=is_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            if codes is not None and codes.shape[-1] > 0:
                # Decode codes to audio
                audio = self.audio_tokenizer.decode(codes)
                audio = audio.squeeze()

                token_count = codes.shape[-1] * codes.shape[1]

                yield self.generate_result(
                    audio=audio,
                    start_time=time_start,
                    token_count=token_count,
                    segment_idx=segment_idx,
                )

    def _generate_codes(
        self,
        input_ids: mx.array,
        is_text: mx.array,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> mx.array:
        """Generate codec tokens autoregressively."""
        B = input_ids.shape[0]
        cache = None
        generated_first_codes = []
        hidden_states_list = []

        # Generate first codebook autoregressively
        for step in range(max_tokens):
            # Forward pass
            logits, cache, hidden = self.talker(
                input_ids, is_text, mask=None, cache=cache
            )

            # Get logits for last position
            next_logits = logits[:, -1, :]

            # Sample next token
            next_token = self._sample(next_logits, temperature, top_p)

            generated_first_codes.append(next_token)
            hidden_states_list.append(hidden[:, -1:, :])

            # Check for EOS
            if mx.all(next_token == self.config.codec_eos_token_id):
                break

            # Prepare next input (single token)
            input_ids = next_token[:, None]
            is_text = mx.zeros(input_ids.shape, dtype=mx.bool_)

            # Evaluate periodically for memory efficiency
            if step % 50 == 0:
                mx.eval(cache)

        if not generated_first_codes:
            return None

        # Stack generated codes
        first_codes = mx.stack(generated_first_codes, axis=1)  # (batch, seq_len)
        hidden_states = mx.concatenate(hidden_states_list, axis=1)

        # Generate remaining codebooks using code predictor
        remaining_codes = self.talker.code_predictor(hidden_states, first_codes)

        # Combine all codebooks: (batch, 16, seq_len)
        all_codes = mx.concatenate([first_codes[:, None, :], remaining_codes], axis=1)

        return all_codes

    def _sample(
        self,
        logits: mx.array,
        temperature: float,
        top_p: float,
    ) -> mx.array:
        """Sample from logits with temperature and top-p."""
        if temperature <= 0:
            return mx.argmax(logits, axis=-1)

        # Apply temperature
        logits = logits / temperature

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
            cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

            # Remove tokens with cumulative prob > top_p
            sorted_mask = cumulative_probs > top_p
            # Keep at least one token
            sorted_mask = mx.concatenate([
                mx.zeros((logits.shape[0], 1), dtype=mx.bool_),
                sorted_mask[:, :-1]
            ], axis=-1)

            sorted_logits = mx.where(sorted_mask, -float('inf'), sorted_logits)

            # Sample from filtered distribution
            probs = mx.softmax(sorted_logits, axis=-1)
            sampled_idx = mx.random.categorical(mx.log(probs + 1e-10))
            token = mx.take_along_axis(sorted_indices, sampled_idx[:, None], axis=-1)[:, 0]
        else:
            probs = mx.softmax(logits, axis=-1)
            token = mx.random.categorical(mx.log(probs + 1e-10))

        return token
