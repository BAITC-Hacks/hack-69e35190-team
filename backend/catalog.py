"""Строгое чтение исходного CSV для фильтрации и карточек."""

import csv
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

from ai_layer.config import DEFAULT_CSV

CALENDAR_START = date(2026, 9, 23)
CALENDAR_END = date(2026, 12, 31)
CITIES = ("Алматы", "Астана", "Зарубежье")
EVENT_TYPES = ("свадьба", "той", "корпоратив", "конференция", "юбилей", "день рождения")
LANGUAGES = ("русский", "казахский", "английский")


@dataclass(frozen=True)
class Contractor:
    id: str
    anon_name: str
    categories: Tuple[str, ...]
    city: str
    price_from_kzt: int
    event_formats: Tuple[str, ...]
    languages: Tuple[str, ...]
    max_hours: Optional[float]
    busy_dates: FrozenSet[date]
    description: str
    synthetic: bool
    city_imputed: bool
    price_imputed: bool


def _boolean(value):
    value = value.strip().lower()
    if value not in ("true", "false"):
        raise ValueError("флаг должен быть True или False")
    return value == "true"


def _list(value):
    items = tuple(item.strip() for item in value.split("|") if item.strip())
    if not items:
        raise ValueError("пустой список")
    return items


def _canonical_labels(profiles):
    """Единые названия внутри каталога; выбор не зависит от порядка строк CSV.

    Исходный CSV и описания не меняются. Нормализуются только метки,
    участвующие в строгих фильтрах, а не имена подрядчиков или их id.
    """
    seeds = {"city": CITIES, "categories": (), "event_formats": EVENT_TYPES,
             "languages": LANGUAGES}
    labels = {}
    for field, preferred in seeds.items():
        choices = {value.casefold(): value for value in preferred}
        values = {value for profile in profiles
                  for value in ((profile.city,) if field == "city" else getattr(profile, field))}
        for value in sorted(values):
            choices.setdefault(value.casefold(), value)
        labels[field] = choices

    normalized = []
    for profile in profiles:
        fields = {"city": labels["city"][profile.city.casefold()]}
        for field in ("categories", "event_formats", "languages"):
            fields[field] = tuple(dict.fromkeys(labels[field][value.casefold()]
                                               for value in getattr(profile, field)))
        normalized.append(replace(profile, **fields))
    return tuple(normalized)


def load_catalog(path=DEFAULT_CSV, calendar_start=CALENDAR_START, calendar_end=CALENDAR_END):
    profiles = []
    seen = set()
    required = set(Contractor.__dataclass_fields__)
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV не содержит все обязательные поля")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Повторяющиеся названия столбцов CSV")
        for line_number, row in enumerate(reader, start=2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("число столбцов не совпадает с заголовком")
                if len(profiles) >= 5000:
                    raise ValueError("лимит — 5000 профилей")
                profile_id = row["id"].strip()
                if not profile_id or profile_id in seen:
                    raise ValueError("пустой или повторяющийся id")
                seen.add(profile_id)
                price = int(row["price_from_kzt"])
                hours = row["max_hours"].strip()
                max_hours = float(hours) if hours.lower() not in ("", "null") else None
                if price < 0 or (max_hours is not None and (not math.isfinite(max_hours) or max_hours <= 0)):
                    raise ValueError("цена или часы вне допустимого диапазона")
                dates = frozenset(date.fromisoformat(item.strip()) for item in row["busy_dates"].split("|") if item.strip())
                if any(day < calendar_start or day > calendar_end for day in dates):
                    raise ValueError("занятая дата вне календаря")
                city = row["city"].strip()
                if not city:
                    raise ValueError("пустой город")
                formats = _list(row["event_formats"])
                languages = _list(row["languages"])
                profile = Contractor(
                    id=profile_id,
                    anon_name=row["anon_name"].strip(),
                    categories=_list(row["categories"]),
                    city=city,
                    price_from_kzt=price,
                    event_formats=formats,
                    languages=languages,
                    max_hours=max_hours,
                    busy_dates=dates,
                    description=row["description"].strip(),
                    synthetic=_boolean(row["synthetic"]),
                    city_imputed=_boolean(row["city_imputed"]),
                    price_imputed=_boolean(row["price_imputed"]),
                )
                if not profile.anon_name or not profile.description:
                    raise ValueError("пустое имя или описание")
                profiles.append(profile)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError("CSV, строка {}: {}".format(line_number, error)) from error
    if not profiles:
        raise ValueError("Каталог пуст")
    return _canonical_labels(profiles)
