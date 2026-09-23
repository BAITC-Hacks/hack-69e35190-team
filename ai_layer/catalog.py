"""Чтение описаний и точных фрагментов исходного CSV."""

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CSV


@dataclass(frozen=True)
class Fragment:
    profile_id: str
    fragment_index: int
    start: int
    end: int
    text: str


def _segments(description):
    start = 0
    for match in re.finditer(r"[.!?](?=\s|$)|[•\n]", description):
        is_separator = match.group() in {"•", "\n"}
        end = match.start() if is_separator else match.end()
        if end > start:
            yield start, end
        start = match.end()
    if start < len(description):
        yield start, len(description)


def _bounded_spans(description, start, end, limit=320):
    while end - start > limit:
        cut = description.rfind(" ", start + 1, start + limit + 1)
        if cut <= start:
            break
        yield start, cut
        start = cut + 1
    yield start, end


def fragments_for(profile_id, description):
    result = []
    for start, end in _segments(description):
        for part_start, part_end in _bounded_spans(description, start, end):
            text = description[part_start:part_end]
            left = len(text) - len(text.lstrip())
            right = len(text.rstrip())
            part_start += left
            part_end = part_start + right - left
            text = description[part_start:part_end]
            if len(text) >= 22 and re.search(r"[А-Яа-яA-Za-z]", text):
                result.append(Fragment(profile_id, len(result), part_start, part_end, text))
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
