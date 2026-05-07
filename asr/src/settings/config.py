import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class Config:
    output_dir: str
    temp_dir: str = "./temp"

    audio_format: str = "mp3"
    max_file_size_mb: int = 25
    max_audio_chunk_minutes: int = 10

    whisper_backend: str = "api"          # "api" or "local"
    whisper_model_name: str = "base"      # local whisper only
    whisper_language: Optional[str] = None

    openai_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path)

        whisper_backend = os.getenv("WHISPER_BACKEND", "api").strip().lower()
        if whisper_backend not in {"api", "local"}:
            raise ValueError(
                f"Invalid WHISPER_BACKEND={whisper_backend}. Must be 'api' or 'local'."
            )

        whisper_model_name = os.getenv("WHISPER_MODEL_NAME", "base").strip()
        whisper_language = os.getenv("WHISPER_LANGUAGE", "").strip() or None
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None

        output_dir = os.getenv("OUTPUT_DIR", "./outputs").strip()
        temp_dir = os.getenv("TEMP_DIR", "./temp").strip()
        audio_format = os.getenv("AUDIO_FORMAT", "mp3").strip()
        max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "25").strip())
        max_audio_chunk_minutes = int(os.getenv("MAX_AUDIO_CHUNK_MINUTES", "10").strip())

        if whisper_backend == "api" and not openai_api_key:
            raise ValueError("WHISPER_BACKEND='api' but OPENAI_API_KEY is not set.")

        return cls(
            output_dir=output_dir,
            temp_dir=temp_dir,
            audio_format=audio_format,
            max_file_size_mb=max_file_size_mb,
            max_audio_chunk_minutes=max_audio_chunk_minutes,
            whisper_backend=whisper_backend,
            whisper_model_name=whisper_model_name,
            whisper_language=whisper_language,
            openai_api_key=openai_api_key,
        )