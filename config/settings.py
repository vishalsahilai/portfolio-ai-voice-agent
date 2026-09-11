from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # Gemini
    GEMINI_API_KEY1: str = ""
    GEMINI_API_KEY2: str = ""
    GEMINI_API_KEY3: str = ""
    GEMINI_API_KEY4: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    @property
    def GEMINI_API_KEYS(self) -> List[str]:
        return [
            key.strip()
            for key in (
                self.GEMINI_API_KEY1,
                self.GEMINI_API_KEY2,
                self.GEMINI_API_KEY3,
                self.GEMINI_API_KEY4,
            )
            if key.strip()
        ]

    # Deepgram STT
    DEEPGRAM_API_KEY1: str = ""
    DEEPGRAM_API_KEY2: str = ""
    DEEPGRAM_API_KEY3: str = ""
    DEEPGRAM_API_KEY4: str = ""

    DEEPGRAM_MODEL: str = "nova-3"
    DEEPGRAM_LANGUAGE: str = "en-US"
    DEEPGRAM_ENDPOINTING_MS: int = 350
    DEEPGRAM_UTTERANCE_END_MS: int = 1000

    @property
    def DEEPGRAM_API_KEYS(self) -> List[str]:
        return [
            key.strip()
            for key in (
                self.DEEPGRAM_API_KEY1,
                self.DEEPGRAM_API_KEY2,
                self.DEEPGRAM_API_KEY3,
                self.DEEPGRAM_API_KEY4,
            )
            if key.strip()
        ]

    # ElevenLabs TTS
    ELEVENLABS_API_KEYS1: str = ""
    ELEVENLABS_VOICE_ID1: str = ""

    ELEVENLABS_API_KEYS2: str = ""
    ELEVENLABS_VOICE_ID2: str = ""

    ELEVENLABS_API_KEYS3: str = ""
    ELEVENLABS_VOICE_ID3: str = ""

    ELEVENLABS_API_KEYS4: str = ""
    ELEVENLABS_VOICE_ID4: str = ""

    ELEVENLABS_API_KEYS5: str = ""
    ELEVENLABS_VOICE_ID5: str = ""

    ELEVENLABS_API_KEYS6: str = ""
    ELEVENLABS_VOICE_ID6: str = ""

    ELEVENLABS_API_KEYS7: str = ""
    ELEVENLABS_VOICE_ID7: str = ""

    ELEVENLABS_API_KEYS8: str = ""
    ELEVENLABS_VOICE_ID8: str = ""

    ELEVENLABS_MODEL_ID: str = "eleven_flash_v2_5"

    @property
    def ELEVENLABS_ACCOUNT_POOL(self) -> List[dict]:
        pairs = (
            (self.ELEVENLABS_API_KEYS1, self.ELEVENLABS_VOICE_ID1),
            (self.ELEVENLABS_API_KEYS2, self.ELEVENLABS_VOICE_ID2),
            (self.ELEVENLABS_API_KEYS3, self.ELEVENLABS_VOICE_ID3),
            (self.ELEVENLABS_API_KEYS4, self.ELEVENLABS_VOICE_ID4),
            (self.ELEVENLABS_API_KEYS5, self.ELEVENLABS_VOICE_ID5),
            (self.ELEVENLABS_API_KEYS6, self.ELEVENLABS_VOICE_ID6),
            (self.ELEVENLABS_API_KEYS7, self.ELEVENLABS_VOICE_ID7),
            (self.ELEVENLABS_API_KEYS8, self.ELEVENLABS_VOICE_ID8),
        )

        return [
            {
                "api_key": api_key.strip(),
                "voice_id": voice_id.strip(),
            }
            for api_key, voice_id in pairs
            if api_key.strip() and voice_id.strip()
        ]

    # Audio
    AUDIO_SAMPLE_RATE: int = 16000

    # Kept temporarily for compatibility with old call/VAD classes.
    AUDIO_CHUNK_MS: int = 30
    SILENCE_THRESHOLD_MS: int = 700

    # Pinecone / RAG
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "voice-agent"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"

    EMBEDDING_MODEL_NAME: str = "llama-text-embed-v2"
    EMBEDDING_DIMENSION: int = 384
    RAG_TOP_K: int = 3

    # MongoDB
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "ai_voice_agent"
    SESSION_EXPIRY_HOURS: int = 2

    @property
    def ALIBABA_MODEL_LIST(self) -> list[str]:
        return [
            model.strip()
            for model in self.ALIBABA_MODELS.split(",")
            if model.strip()
        ]

    ALIBABA_API_KEY: str = ""

    ALIBABA_BASE_URL: str = (
        "https://dashscope-intl.aliyuncs.com/"
        "compatible-mode/v1"
    )

    ALIBABA_MODELS: str = (
        "qwen3.6-flash,"
        "qwen3.5-flash,"
        "qwen-flash"
    )

    ALIBABA_TIMEOUT_SECONDS: float = 5.0

    ALIBABA_TRANSIENT_COOLDOWN_SECONDS: float = 30.0
    ALIBABA_RATE_LIMIT_COOLDOWN_SECONDS: float = 60.0
    ALIBABA_ACCESS_COOLDOWN_SECONDS: float = 300.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()