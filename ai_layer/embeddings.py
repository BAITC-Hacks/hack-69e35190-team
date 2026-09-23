"""Минимальный HTTP-клиент эмбеддингов OpenAI без внешних зависимостей."""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    pass


class OpenAIEmbeddings:
    endpoint = "https://api.openai.com/v1/embeddings"

    def __init__(self, api_key, model="text-embedding-3-small", dimensions=256, timeout=3):
        if not api_key:
            raise ValueError("OPENAI_API_KEY не задан")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.timeout = timeout

    def embed(self, texts):
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": texts,
                              "dimensions": self.dimensions, "encoding_format": "float"}).encode("utf-8")
        request = Request(self.endpoint, data=payload, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.api_key,
        })
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
            indexed = {item["index"]: item["embedding"] for item in data["data"]}
            vectors = [indexed[i] for i in range(len(texts))]
            if any(len(vector) != self.dimensions or
                   any(not isinstance(n, (int, float)) or not math.isfinite(n) for n in vector)
                   for vector in vectors):
                raise ValueError("Неверный размер или значения вектора")
            return vectors
        except HTTPError as error:
            raise EmbeddingError("Сервис эмбеддингов вернул HTTP {}".format(error.code)) from None
        except (URLError, TimeoutError, OSError) as error:
            raise EmbeddingError("Сервис эмбеддингов недоступен") from None
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EmbeddingError("Некорректный ответ сервиса эмбеддингов") from None
