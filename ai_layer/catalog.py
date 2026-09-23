"""Чтение описаний и точных фрагментов исходного CSV."""

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CSV


# Изменение границ делает старые векторы непригодными, даже при том же CSV.
FRAGMENTATION_VERSION = 3
_SECTION_START = re.compile(
    r"(?<!\w)(?:(?:Расширенный|Большой|Музыкальный|Стандартный|Основной)\s+состав\b"
    r"|Репертуар\s*(?::|включает\b)|Языки\s+ведения\s*:|Крупные\s+проекты\b)"
)
_ABBREVIATIONS = {"г", "ул", "д", "им", "др", "пр", "т", "п", "ч", "чел", "кв", "м", "см"}


def _is_sentence_end(description, position):
    char = description[position]
    if char not in ".!?":
        return False
    if position + 1 == len(description) or description[position + 1].isspace():
        return True
    # В исходном CSV встречается «атмосферу.Команда» без пробела. Не
    # объединяем два предложения, но не режем инициалы «И.Иванов»/«ул.Абая».
    if not description[position + 1].isupper():
        return False
    if char in "!?":
        return True
    previous = re.search(r"([А-Яа-яA-Za-z]+)$", description[:position])
    return bool(previous and len(previous.group(1)) > 1 and
                previous.group(1).casefold() not in _ABBREVIATIONS)


@dataclass(frozen=True)
class Fragment:
    profile_id: str
    fragment_index: int
    start: int
    end: int
    text: str


def _segments(description):
    """Целые предложения и явно озаглавленные блоки, с точными смещениями.

    Длина — не граница мысли: нельзя начинать цитату с «музыканта)» или
    завершать её посреди имени. Длинный блок без естественной границы
    остаётся целым; отделяем только предложения, bullets и заголовки списков.
    """
    start = 0
    brackets = []
    section_starts = {match.start() for match in _SECTION_START.finditer(description)}
    for position, char in enumerate(description):
        if char in "([{":
            brackets.append(char)
        elif char in ")]}":
            if brackets:
                brackets.pop()
        if brackets:
            continue
        if position in section_starts and position > start:
            yield start, position
            start = position
        is_separator = char in {"•", "\n"}
        is_sentence_end = _is_sentence_end(description, position)
        if not is_separator and not is_sentence_end:
            continue
        end = position if is_separator else position + 1
        if end > start:
            yield start, end
        start = position + 1
    if start < len(description):
        yield start, len(description)


def fragments_for(profile_id, description):
    result = []
    for start, end in _segments(description):
        text = description[start:end]
        left = len(text) - len(text.lstrip())
        right = len(text.rstrip())
        start += left
        end = start + right - left
        text = description[start:end]
        if len(text) >= 22 and re.search(r"[А-Яа-яA-Za-z]", text):
            result.append(Fragment(profile_id, len(result), start, end, text))
    if not result and description.strip():
        start = len(description) - len(description.lstrip())
        end = len(description.rstrip())
        result.append(Fragment(profile_id, 0, start, end, description[start:end]))
    return tuple(result)


class Catalog:
    def __init__(self, path=DEFAULT_CSV):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.profiles = {}
        self.fragments = {}
        with self.path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                row = {key: value.strip() for key, value in row.items()}
                if row["max_hours"].lower() == "null":
                    row["max_hours"] = ""
                profile_id = row["id"].strip()
                if not profile_id or profile_id in self.profiles:
                    raise ValueError("Пустой или повторяющийся id в CSV")
                self.profiles[profile_id] = row
                self.fragments[profile_id] = fragments_for(profile_id, row["description"])

    def all_fragments(self):
        return tuple(fragment for profile_id in sorted(self.fragments)
                     for fragment in self.fragments[profile_id])
