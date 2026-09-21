from typing import List
 
from config.settings import settings

class FrameBuffer:
    def __init__(self, frame_bytes: int = None):
        # 16-bit mono PCM: bytes_per_frame = sample_rate * (frame_ms/1000) * 2
        self.frame_bytes = frame_bytes or int(
            settings.AUDIO_SAMPLE_RATE * (settings.AUDIO_CHUNK_MS / 1000.0) * 2
        )
        self._buffer = bytearray()
 
    def push(self, chunk: bytes) -> List[bytes]:
        """Add incoming bytes, return however many complete frames can now be sliced off."""
        self._buffer.extend(chunk)
        frames = []
        while len(self._buffer) >= self.frame_bytes:
            frames.append(bytes(self._buffer[: self.frame_bytes]))
            del self._buffer[: self.frame_bytes]
        return frames
 
    def reset(self) -> None:
        self._buffer.clear()

class UtteranceBuffer:
    def __init__(self):
        self._chunks: List[bytes] = []
 
    def add(self, frame: bytes) -> None:
        self._chunks.append(frame)
 
    def get_audio(self) -> bytes:
        return b"".join(self._chunks)
 
    def duration_seconds(self) -> float:
        total_bytes = sum(len(c) for c in self._chunks)
        # 16-bit mono PCM = 2 bytes per sample
        return total_bytes / 2 / settings.AUDIO_SAMPLE_RATE
 
    def reset(self) -> None:
        self._chunks.clear()