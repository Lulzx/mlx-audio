# Copyright (c) 2025 MLX Audio Contributors
# Licensed under the MIT License

"""Qwen3-TTS Model implementation for MLX.

This implementation uses the official qwen-tts package for inference,
as the model architecture is complex with a talker module, code predictor,
and multi-head outputs.

Requires: pip install qwen-tts torch
A native MLX implementation is planned for a future release.
"""

import time
from pathlib import Path
from typing import Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import GenerationResult
from .config import ModelConfig


class Model(nn.Module):
    """Qwen3-TTS Model for text-to-speech synthesis.

    This model uses the official qwen-tts package for inference,
    providing voice cloning, preset speakers, and voice design capabilities.

    Supports three model variants:
    - Base: Voice cloning with reference audio
    - CustomVoice: Preset speakers with instruction-based emotion control
    - VoiceDesign: Natural language voice description
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.tokenizer = None
        self._qwen_model = None
        self._model_path = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.config.position_id_per_seconds

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Hook called after model weights are loaded.

        Stores the model path for lazy loading of the qwen-tts model.
        """
        model._model_path = str(model_path)
        return model

    def sanitize(self, weights: dict) -> dict:
        """Sanitize weights - skip all weights since we use qwen-tts backend.

        This model uses the official qwen-tts package for inference,
        so we don't need to load any MLX weights.
        """
        return {}

    def _load_qwen_model(self):
        """Load the qwen-tts model."""
        if self._qwen_model is not None:
            return

        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            model_path = self._model_path or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

            # Determine device
            if torch.cuda.is_available():
                device = "cuda:0"
                dtype = torch.bfloat16
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32  # MPS doesn't support bfloat16
            else:
                device = "cpu"
                dtype = torch.float32

            self._qwen_model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=device,
                dtype=dtype,
            )

        except ImportError as e:
            raise ImportError(
                "Qwen3-TTS requires the qwen-tts package. "
                "Install with: pip install qwen-tts torch"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen3-TTS model: {e}. "
                "Make sure you have internet access and the model is available."
            ) from e

    def get_language_id(self, language: str) -> int:
        """Get the language token ID."""
        lang_lower = language.lower().replace(" ", "_")
        return self.config.codec_language_id.get(
            lang_lower, self.config.codec_language_id["english"]
        )

    def get_speaker_id(self, speaker: str) -> int:
        """Get the speaker token ID."""
        spk_lower = speaker.lower().replace(" ", "_")
        return self.config.spk_id.get(spk_lower)

    def generate_result(
        self,
        audio: mx.array,
        start_time: float,
        token_count: int,
        segment_idx: int,
        **kwargs,
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
                "tokens-per-sec": round(token_count / elapsed_time, 2)
                if elapsed_time > 0
                else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed_time, 2)
                if elapsed_time > 0
                else 0,
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
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
        split_pattern: str = "\n",
        stream: bool = False,
        streaming_interval: float = 2.0,
        verbose: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text.

        This method supports three generation modes based on provided arguments:
        1. Voice cloning: ref_audio + ref_text provided (requires Base model)
        2. Custom voice: voice (speaker name) provided (requires CustomVoice model)
        3. Voice design: instruct provided without voice (requires VoiceDesign model)

        Args:
            text: Text to synthesize
            voice: Speaker name for CustomVoice model (e.g., "vivian", "ryan")
            ref_audio: Reference audio for voice cloning (as mx.array or numpy)
            ref_text: Transcript of reference audio
            instruct: Instruction for emotion/style control or voice description
            language: Target language (e.g., "english", "chinese")
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            max_tokens: Maximum tokens to generate
            split_pattern: Pattern to split text into segments
            stream: Whether to stream partial results (not fully supported yet)
            streaming_interval: Interval (in seconds) between streaming updates
            verbose: Whether to show progress

        Yields:
            GenerationResult objects containing audio and metadata
        """
        import numpy as np

        self._load_qwen_model()

        # Split text into segments
        prompt_text = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = [p for p in prompt_text.split(split_pattern) if p.strip()]

        # Capitalize language for qwen-tts API
        lang_capitalized = language.capitalize()

        for segment_idx, segment_text in enumerate(prompts):
            time_start = time.perf_counter()

            if verbose:
                print(
                    f"Generating segment {segment_idx + 1}/{len(prompts)}: {segment_text[:50]}..."
                )

            try:
                # Determine generation mode and call appropriate method
                if ref_audio is not None and ref_text is not None:
                    # Voice cloning mode (Base model)
                    audio_output, sr = self._generate_with_cloning(
                        text=segment_text,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        language=lang_capitalized,
                    )

                elif voice is not None:
                    # CustomVoice mode
                    audio_output, sr = self._generate_with_speaker(
                        text=segment_text,
                        speaker=voice,
                        instruct=instruct or "",
                        language=lang_capitalized,
                    )

                elif instruct is not None:
                    # VoiceDesign mode
                    audio_output, sr = self._generate_with_design(
                        text=segment_text,
                        instruct=instruct,
                        language=lang_capitalized,
                    )

                else:
                    raise ValueError(
                        "Must provide either: "
                        "(1) ref_audio + ref_text for voice cloning, "
                        "(2) voice for custom voice, or "
                        "(3) instruct for voice design"
                    )

                # Convert output to MLX array
                if audio_output is not None:
                    if hasattr(audio_output, "cpu"):
                        # PyTorch tensor
                        audio_np = audio_output.cpu().numpy()
                    elif isinstance(audio_output, list):
                        # List of arrays (batch output)
                        audio_np = np.array(audio_output[0])
                    else:
                        audio_np = np.array(audio_output)

                    # Flatten if needed
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()

                    audio_mx = mx.array(audio_np)

                    # Calculate token count estimate
                    token_count = int(
                        audio_mx.shape[-1]
                        / sr
                        * self.frame_rate
                        * self.config.num_code_groups
                    )

                    yield self.generate_result(
                        audio=audio_mx,
                        start_time=time_start,
                        token_count=token_count,
                        segment_idx=segment_idx,
                    )

            except Exception as e:
                if verbose:
                    print(f"Error generating segment {segment_idx}: {e}")
                raise

    def _generate_with_cloning(
        self,
        text: str,
        ref_audio,
        ref_text: str,
        language: str,
    ):
        """Generate speech using voice cloning (requires Base model)."""
        import numpy as np

        # Convert ref_audio to numpy if needed
        if isinstance(ref_audio, mx.array):
            ref_audio_np = np.array(ref_audio)
        else:
            ref_audio_np = np.array(ref_audio)

        # Check if model supports voice cloning
        if hasattr(self._qwen_model, "generate"):
            # Base model with voice cloning
            wavs, sr = self._qwen_model.generate(
                text=text,
                ref_audio=ref_audio_np,
                ref_text=ref_text,
                language=language,
            )
            return wavs, sr
        else:
            raise NotImplementedError(
                "Voice cloning requires a Base model variant. "
                "Try loading 'Qwen/Qwen3-TTS-12Hz-0.6B-Base' or '1.7B-Base'."
            )

    def _generate_with_speaker(
        self,
        text: str,
        speaker: str,
        instruct: str,
        language: str,
    ):
        """Generate speech using a preset speaker (CustomVoice model)."""
        # Validate speaker
        spk_title = speaker.title().replace("_", "_")  # Keep underscores
        # Map common name formats
        speaker_map = {
            "vivian": "Vivian",
            "serena": "Serena",
            "uncle_fu": "Uncle_Fu",
            "dylan": "Dylan",
            "eric": "Eric",
            "ryan": "Ryan",
            "aiden": "Aiden",
            "ono_anna": "Ono_Anna",
            "sohee": "Sohee",
        }
        speaker_name = speaker_map.get(speaker.lower(), spk_title)

        # Check if model has custom voice generation
        if hasattr(self._qwen_model, "generate_custom_voice"):
            wavs, sr = self._qwen_model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker_name,
                instruct=instruct,
            )
            return wavs, sr
        else:
            raise NotImplementedError(
                "Custom voice generation requires a CustomVoice model variant. "
                "Try loading 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice' or '1.7B-CustomVoice'."
            )

    def _generate_with_design(
        self,
        text: str,
        instruct: str,
        language: str,
    ):
        """Generate speech using voice design (VoiceDesign model)."""
        # Check if model has voice design generation
        if hasattr(self._qwen_model, "generate_voice_design"):
            wavs, sr = self._qwen_model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
            return wavs, sr
        else:
            raise NotImplementedError(
                "Voice design generation requires a VoiceDesign model variant. "
                "Try loading 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'."
            )
