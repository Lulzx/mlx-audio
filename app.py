"""Streamlit app for testing Qwen3-TTS model."""

import io
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Qwen3-TTS Demo",
    page_icon="🎤",
    layout="centered",
)

st.title("🎤 Qwen3-TTS Demo")
st.markdown("Generate speech with preset voices and emotion control")


@st.cache_resource
def load_model():
    """Load the Qwen3-TTS model (cached)."""
    from mlx_audio.tts import load
    return load("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")


# Sidebar settings
with st.sidebar:
    st.header("Settings")

    speaker = st.selectbox(
        "Speaker",
        options=["Ryan", "Vivian", "Serena", "Aiden", "Dylan", "Eric", "Uncle_Fu", "Ono_Anna", "Sohee"],
        index=0,
    )

    language = st.selectbox(
        "Language",
        options=["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Italian", "Portuguese", "Russian"],
        index=0,
    )

    instruct = st.text_input(
        "Emotion/Style (optional)",
        placeholder="e.g., happy and energetic, calm and soothing",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more deterministic, Higher = more varied",
    )

    st.markdown("---")
    st.markdown("**Speaker Info:**")
    speaker_info = {
        "Ryan": "🇺🇸 English male",
        "Vivian": "🇨🇳 Chinese female",
        "Serena": "🇨🇳 Chinese female",
        "Aiden": "🇺🇸 English male",
        "Dylan": "🇨🇳 Beijing dialect male",
        "Eric": "🇨🇳 Sichuan dialect male",
        "Uncle_Fu": "🇨🇳 Chinese male",
        "Ono_Anna": "🇯🇵 Japanese female",
        "Sohee": "🇰🇷 Korean female",
    }
    st.info(speaker_info.get(speaker, ""))

# Main content
text = st.text_area(
    "Enter text to synthesize",
    value="Hello! Welcome to the Qwen3 text to speech demo. This model supports multiple languages and emotion control.",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    generate_btn = st.button("🎵 Generate", type="primary", use_container_width=True)

if generate_btn and text.strip():
    with st.spinner("Loading model..."):
        model = load_model()

    with st.spinner("Generating speech..."):
        try:
            # Generate audio
            for result in model.generate(
                text=text,
                voice=speaker.lower(),
                language=language,
                instruct=instruct if instruct else None,
                temperature=temperature,
            ):
                audio = np.array(result.audio)
                sample_rate = result.sample_rate

                # Convert to int16 for WAV
                audio_int16 = (audio * 32767).astype(np.int16)

                # Create WAV bytes
                import wave
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(audio_int16.tobytes())
                buffer.seek(0)

                # Display results
                st.success(f"Generated {result.audio_duration} of audio")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", result.audio_duration)
                with col2:
                    st.metric("Sample Rate", f"{sample_rate} Hz")
                with col3:
                    st.metric("RTF", f"{result.real_time_factor:.2f}x")

                # Audio player
                st.audio(buffer, format="audio/wav")

                # Download button
                buffer.seek(0)
                st.download_button(
                    label="📥 Download WAV",
                    data=buffer,
                    file_name="qwen3_tts_output.wav",
                    mime="audio/wav",
                )

        except Exception as e:
            st.error(f"Error generating speech: {e}")

elif generate_btn:
    st.warning("Please enter some text to synthesize.")

# Footer
st.markdown("---")
st.markdown(
    "<small>Powered by [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) and [MLX Audio](https://github.com/Blaizzy/mlx-audio)</small>",
    unsafe_allow_html=True,
)
