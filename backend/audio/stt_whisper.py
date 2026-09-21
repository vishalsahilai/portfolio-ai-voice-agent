"""
audio/stt_whisper.py

Local Whisper STT using faster-whisper.
vad_filter=True tells Whisper to internally skip silence segments,
which prevents hallucinations like "you", "bye", "thanks" on near-silent audio.
"""

import numpy as np
from faster_whisper import WhisperModel

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class WhisperSTT:
    def __init__(self, model_size: str = None, device: str = None):
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.device = device or settings.WHISPER_DEVICE
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info(
                f"Loading faster-whisper model '{self.model_size}' on {self.device} (first call only)..."
            )
            compute_type = "float16" if self.device == "cuda" else "int8"
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
            logger.info("faster-whisper model loaded")

    @staticmethod
    def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """Convert raw 16-bit PCM bytes to float32 in [-1, 1] for faster-whisper."""
        int16_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_samples.astype(np.float32) / 32768.0

    def transcribe(self, pcm_bytes: bytes, language: str = "en") -> str:
        """
        pcm_bytes : raw 16-bit mono PCM at AUDIO_SAMPLE_RATE (16 kHz).
        Returns the transcribed text, or "" if nothing was detected.

        vad_filter=True  — Whisper internally skips silence; prevents
                           hallucinations ("you", "bye") on near-silent clips.
        no_speech_threshold — if Whisper's own no-speech probability exceeds
                           this, the segment is discarded.
        """
        if not pcm_bytes:
            return ""

        self._ensure_loaded()
        audio = self.pcm_bytes_to_float32(pcm_bytes)

        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,            # ← eliminates most hallucinations
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
            no_speech_threshold=0.6,    # discard low-confidence no-speech segments
            log_prob_threshold=-1.0,    # discard very low probability output
        )

        text = " ".join(segment.text for segment in segments).strip()
        logger.info(f"Whisper transcribed: '{text}'")
        return text


whisper_stt = WhisperSTT()
