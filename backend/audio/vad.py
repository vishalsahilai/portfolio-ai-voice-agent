"""
audio/vad.py

Voice Activity Detector wrapping webrtcvad.

Two-layer noise rejection:
  1. RMS energy gate  — frames below a minimum energy level are treated as
     silence regardless of what webrtcvad says. This kills Mac fan noise,
     air conditioning, and mic hiss that webrtcvad occasionally classifies
     as speech.
  2. Consecutive-frame confirmation — N consecutive speech frames (after
     the energy gate) must be seen before we declare "speech_started".
     This eliminates single-frame pops and clicks.
"""

import collections
import math
from typing import Optional

import webrtcvad

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class VoiceActivityDetector:
    def __init__(
        self,
        aggressiveness: int = 3,
        sample_rate: int = None,
        frame_ms: int = None,
        speech_confirm_frames: int = 8,   # raised from 5 → 240ms of confirmed speech required
        silence_confirm_ms: int = None,
        min_rms_energy: float = 150.0,    # RMS gate: below this = always silence
    ):
        """
        aggressiveness       : 0-3 (3 = most aggressive noise filtering)
        speech_confirm_frames: consecutive speech frames needed to trigger
                               speech_started (each frame is frame_ms long).
                               8 frames × 30ms = 240ms — eliminates mic pops.
        silence_confirm_ms   : silence duration before speech_ended fires.
        min_rms_energy       : RMS amplitude gate (0-32768 scale for 16-bit PCM).
                               Frames with RMS below this are treated as silence
                               regardless of webrtcvad's opinion.
                               150 ≈ -46 dBFS — well above hiss, below soft speech.
        """
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("aggressiveness must be 0-3")

        self.sample_rate = sample_rate or settings.AUDIO_SAMPLE_RATE
        self.frame_ms = frame_ms or settings.AUDIO_CHUNK_MS
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 (webrtcvad requirement)")

        self._vad = webrtcvad.Vad(aggressiveness)
        self._bytes_per_frame = int(
            self.sample_rate * (self.frame_ms / 1000.0) * 2
        )  # 16-bit = 2 bytes/sample

        self.speech_confirm_frames = speech_confirm_frames
        self.silence_confirm_frames = int(
            (silence_confirm_ms or settings.SILENCE_THRESHOLD_MS) / self.frame_ms
        )
        self.min_rms_energy = min_rms_energy

        # Rolling state
        self._recent_speech_flags = collections.deque(maxlen=speech_confirm_frames)
        self._consecutive_silence_frames = 0
        self.is_speaking = False

    @staticmethod
    def _rms(frame: bytes) -> float:
        """Compute RMS energy of a 16-bit PCM frame."""
        if not frame:
            return 0.0
        # Interpret bytes as signed 16-bit samples
        n = len(frame) // 2
        total = 0
        for i in range(n):
            sample = int.from_bytes(frame[i * 2: i * 2 + 2], "little", signed=True)
            total += sample * sample
        return math.sqrt(total / n) if n else 0.0

    def is_speech_frame(self, frame: bytes) -> bool:
        """
        Two-layer check:
          1. RMS energy gate — low-energy frames are always silence.
          2. webrtcvad — classifier on frames that pass the energy gate.
        """
        if len(frame) != self._bytes_per_frame:
            raise ValueError(
                f"Expected {self._bytes_per_frame} bytes for a {self.frame_ms}ms frame "
                f"at {self.sample_rate}Hz, got {len(frame)}"
            )

        # Layer 1: energy gate
        if self._rms(frame) < self.min_rms_energy:
            return False

        # Layer 2: webrtcvad
        return self._vad.is_speech(frame, self.sample_rate)

    def process_frame(self, frame: bytes) -> Optional[str]:
        """
        Feed one frame at a time. Returns:
          "speech_started" — user just started talking (N confirmed frames)
          "speech_ended"   — user stopped talking (silence_confirm_ms of silence)
          None             — no state change, keep buffering
        """
        speech = self.is_speech_frame(frame)
        self._recent_speech_flags.append(speech)

        if not self.is_speaking:
            # Require ALL confirm frames to be speech before triggering
            if (
                len(self._recent_speech_flags) == self.speech_confirm_frames
                and all(self._recent_speech_flags)
            ):
                self.is_speaking = True
                self._consecutive_silence_frames = 0
                logger.info("VAD: speech_started")
                return "speech_started"
            return None

        # Already speaking — watch for silence
        if speech:
            self._consecutive_silence_frames = 0
        else:
            self._consecutive_silence_frames += 1
            if self._consecutive_silence_frames >= self.silence_confirm_frames:
                self.is_speaking = False
                self._recent_speech_flags.clear()
                logger.info("VAD: speech_ended")
                return "speech_ended"

        return None

    def reset(self) -> None:
        self._recent_speech_flags.clear()
        self._consecutive_silence_frames = 0
        self.is_speaking = False
