"""Чтение локального файла ключей; ключи никогда не выводятся в лог."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "datasets/hackathon dataset anonymized .csv"
DEFAULT_INDEX = ROOT / ".cache/description-embeddings.json"


def _dotenv(path):
    values = {}
    if not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.isidentifier():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 256
    chat_model: str = "gpt-4.1-mini-2025-04-14"

    @classmethod
    def load(cls, env_file=ROOT / ".env"):
        file_values = _dotenv(env_file)

        def get(key, default=""):
            return os.environ.get(key, file_values.get(key, default)).strip()

        model = get("AI_EMBEDDING_MODEL", "text-embedding-3-small")
        dimensions = int(get("AI_EMBEDDING_DIMENSIONS", "256"))
        if not model or dimensions <= 0:
            raise ValueError("Неверная настройка модели эмбеддингов")
        return cls(openai_api_key=get("OPENAI_API_KEY"), model=model, dimensions=dimensions,
                   chat_model=get("AI_CHAT_MODEL", "gpt-4.1-mini-2025-04-14"))
