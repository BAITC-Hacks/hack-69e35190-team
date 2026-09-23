"""Поиск внутри уже выбранного профиля: эмбеддинги или локальный запасной путь."""

import json
import math
import os
import re
import tempfile
from pathlib import Path

from .config import DEFAULT_INDEX
from .embeddings import EmbeddingError


EVENT_PATTERNS = {
    "корпоратив": (r"корпорат", r"делов", r"бизнес", r"компан", r"бренд", r"форум"),
    "свадьба": (r"свад", r"церемон", r"невест", r"букет", r"жених"),
    "той": (r"\bтой\b", r"традиц", r"национал"),
    "конференция": (r"конферен", r"форум", r"делов", r"презентац", r"бизнес"),
    "юбилей": (r"юбиле", r"торжеств", r"праздн"),
    "день рождения": (r"рожден", r"праздн", r"вечерин"),
}
CATEGORY_PATTERNS = {
    "Ведущий": (r"ведущ", r"ведени", r"сценар"),
    "Флорист": (r"флорист", r"цвет", r"композиц"),
    "Банкетный зал": (r"зал", r"террас", r"гост", r"банкет"),
    "Фотограф": (r"фото", r"съём", r"кадр"),
}
DISTINCTIVE_PATTERNS = (
    r"танц", r"развлеч", r"реч", r"юмор", r"импровиз", r"традиц",
    r"панорам", r"гор", r"кейтер", r"парков", r"авторск", r"сценар",
    r"делов", r"форум", r"двуязыч", r"букет", r"сезонн", r"интерактив",
    r"итальянск", r"казахск", r"конкурс", r"вместим",
)


def _lexical_score(fragment, order, anon_name):
    text = fragment.text.casefold()
    score = 0.0
    for pattern in EVENT_PATTERNS.get(order["event_type"].casefold(), (r"" + re.escape(order["event_type"].casefold()[:5]),)):
        if re.search(pattern, text):
            score += 6
    for pattern in CATEGORY_PATTERNS.get(order["category"], (re.escape(order["category"].casefold()[:5]),)):
        if re.search(pattern, text):
            score += 2
    score += sum(3.5 for pattern in DISTINCTIVE_PATTERNS if re.search(pattern, text))
    if "сценар" in text and re.search(r"оригинальн|разработ|авторск", text):
        score += 4
    score += min(len(re.findall(r"[А-Яа-яA-Za-z]{4,}", text)), 16) * 0.06
    if anon_name and anon_name.casefold() in text:
        score -= 12
    if re.search(r"отличн(ый|ая) выбор|профессионал сво(его|ей) дела", text):
        score -= 6
    return score


def _cosine(left, right):
    if len(left) != len(right):
        return -1.0
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if not norm_left or not norm_right:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (norm_left * norm_right)


def _search_text(order):
    parts = [order["category"], "для", order["event_type"], "в", order["city"]]
    if order.get("language"):
        parts.extend(("язык", order["language"]))
    if order.get("duration_hours") is not None:
        parts.extend(("длительность", str(order["duration_hours"]), "часов"))
    return " ".join(parts)


class EvidenceSelector:
    def __init__(self, catalog, embedder=None, model=None, dimensions=256, index_path=DEFAULT_INDEX):
        self.catalog = catalog
        self.embedder = embedder
        self.model = model
        self.dimensions = dimensions
        self.index_path = Path(index_path)
        self.fragments = catalog.all_fragments()
        self.vectors = self._load_index()
        self.query_cache = {}

    def _load_index(self):
        if self.embedder is None or not self.index_path.is_file():
            return None
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if (data["schema"] != 1 or data["csv_sha256"] != self.catalog.digest or
                data["model"] != self.model or data["dimensions"] != self.dimensions or
                len(data["vectors"]) != len(self.fragments)):
                return None
            if any(len(vector) != self.dimensions for vector in data["vectors"]):
                return None
            return data["vectors"]
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def prepare_index(self, batch_size=48):
        """Явная подготовка; никогда не вызывается автоматически на пользовательском запросе."""
        if self.embedder is None:
            raise ValueError("Для подготовки индекса нужен OPENAI_API_KEY")
        if batch_size <= 0:
            raise ValueError("batch_size должен быть положительным")
        vectors = []
        for start in range(0, len(self.fragments), batch_size):
            batch = self.fragments[start:start + batch_size]
            vectors.extend(self.embedder.embed([" ".join(item.text.split()) for item in batch]))
        if len(vectors) != len(self.fragments):
            raise EmbeddingError("Индекс эмбеддингов неполный")
        data = {"schema": 1, "csv_sha256": self.catalog.digest, "model": self.model,
                "dimensions": self.dimensions, "vectors": vectors}
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(self.index_path.parent),
                                             prefix=".embedding-", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
            os.replace(temporary, self.index_path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self.vectors = vectors
        return len(self.fragments)

    def select(self, order, profile_id):
        profile = self.catalog.profiles[profile_id]
        candidates = self.catalog.fragments[profile_id]
        if not candidates:
            return None
        lexical = {fragment.fragment_index: _lexical_score(fragment, order, profile["anon_name"])
                   for fragment in candidates}
        vectors = None
        if self.vectors is not None and self.embedder is not None:
            query = _search_text(order)
            if query not in self.query_cache:
                try:
                    self.query_cache[query] = self.embedder.embed([query])[0]
                except (EmbeddingError, IndexError):
                    self.query_cache[query] = None
            vectors = self.query_cache[query]
        if vectors is not None:
            positions = {(item.profile_id, item.fragment_index): i for i, item in enumerate(self.fragments)}
            return max(candidates, key=lambda item: (
                _cosine(vectors, self.vectors[positions[(profile_id, item.fragment_index)]])
                + min(lexical[item.fragment_index], 20) * 0.015,
                -item.fragment_index,
            ))
        return max(candidates, key=lambda item: (lexical[item.fragment_index], -item.fragment_index))
