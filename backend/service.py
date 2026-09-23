"""Валидация, жёсткие фильтры и детерминированный порядок."""

import math
import re
from collections import Counter
from datetime import date

from ai_layer import AIExplainer
from ai_layer.config import DEFAULT_CSV

from .catalog import CALENDAR_END, CALENDAR_START, CITIES, EVENT_TYPES, LANGUAGES, load_catalog

REASON_NAMES = {
    "busy": "заняты на дату",
    "budget": "цена «от» выше бюджета",
    "format": "не берут формат",
    "language": "не работают на запрошенном языке",
    "duration": "не покрывают длительность",
}


def _canonical(value, choices, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{}: требуется непустая строка".format(field))
    text = value.strip().casefold()
    for choice in choices:
        if choice.casefold() == text:
            return choice
    raise ValueError("{}: неизвестное значение".format(field))


def validate_order(data, cities=CITIES, event_types=EVENT_TYPES, languages=LANGUAGES,
                   calendar_start=CALENDAR_START, calendar_end=CALENDAR_END):
    if not isinstance(data, dict):
        raise ValueError("Запрос должен быть JSON-объектом")
    required = {"city", "event_date", "event_type", "category", "budget_kzt"}
    allowed = required | {"language", "duration_hours"}
    if required - data.keys():
        raise ValueError("Не хватает полей: " + ", ".join(sorted(required - data.keys())))
    if data.keys() - allowed:
        raise ValueError("Неизвестные поля: " + ", ".join(sorted(data.keys() - allowed)))
    result = {
        "city": _canonical(data["city"], cities, "city"),
        "event_type": _canonical(data["event_type"], event_types, "event_type"),
    }
    category = data["category"]
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category: требуется непустая строка")
    result["category"] = category.strip()
    raw_date = data["event_date"]
    if not isinstance(raw_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
        raise ValueError("event_date: нужен формат YYYY-MM-DD")
    try:
        event_date = date.fromisoformat(raw_date)
    except ValueError:
        raise ValueError("event_date: такой даты не существует") from None
    if not calendar_start <= event_date <= calendar_end:
        raise ValueError("Календарь известен только с {} по {}".format(calendar_start, calendar_end))
    result["event_date"] = raw_date
    budget = data["budget_kzt"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget_kzt: требуется положительное целое число")
    result["budget_kzt"] = budget
    language = data.get("language")
    if language is not None:
        result["language"] = _canonical(language, languages, "language")
    duration = data.get("duration_hours")
    if duration is not None:
        if (isinstance(duration, bool) or not isinstance(duration, (int, float)) or
                not math.isfinite(duration) or duration <= 0):
            raise ValueError("duration_hours: требуется положительное конечное число")
        result["duration_hours"] = duration
    return result


def _reasons(profile, order, event_date):
    reasons = []
    if event_date in profile.busy_dates:
        reasons.append("busy")
    if profile.price_from_kzt > order["budget_kzt"]:
        reasons.append("budget")
    if order["event_type"] not in profile.event_formats:
        reasons.append("format")
    if order.get("language") and order["language"] not in profile.languages:
        reasons.append("language")
    if (order.get("duration_hours") is not None and profile.max_hours is not None and
            order["duration_hours"] > profile.max_hours):
        reasons.append("duration")
    return reasons


def _card(profile, category):
    return {
        "id": profile.id,
        "anon_name": profile.anon_name,
        "category": category,
        "city": profile.city,
        "price_from_kzt": profile.price_from_kzt,
        "synthetic": profile.synthetic,
        "city_imputed": profile.city_imputed,
        "price_imputed": profile.price_imputed,
    }


def _reason_summary(counts, event_date):
    details = ["{} — {}".format(REASON_NAMES[key], counts[key]) for key in REASON_NAMES if counts[key]]
    return " На {}: {}. Один профиль может иметь несколько причин.".format(
        event_date, "; ".join(details)) if details else ""


class RecommendationService:
    def __init__(self, catalog_csv=DEFAULT_CSV, explainer=None,
                 calendar_start=CALENDAR_START, calendar_end=CALENDAR_END):
        self.calendar_start, self.calendar_end = calendar_start, calendar_end
        self.profiles = load_catalog(catalog_csv, calendar_start, calendar_end)
        self.cities = tuple(sorted(set(CITIES) | {p.city for p in self.profiles}))
        self.event_types = tuple(sorted(set(EVENT_TYPES) | {v for p in self.profiles for v in p.event_formats}))
        self.languages = tuple(sorted(set(LANGUAGES) | {v for p in self.profiles for v in p.languages}))
        self.categories = {category.casefold(): category for profile in self.profiles
                           for category in profile.categories}
        self.explainer = explainer if explainer is not None else AIExplainer(catalog_csv=catalog_csv)

    def validate(self, data):
        return validate_order(data, self.cities, self.event_types, self.languages,
                              self.calendar_start, self.calendar_end)

    def recommend(self, data):
        order = self.validate(data)
        category = self.categories.get(order["category"].casefold(), order["category"])
        order["category"] = category
        event_date = date.fromisoformat(order["event_date"])
        candidates = [profile for profile in self.profiles
                      if profile.city == order["city"] and category in profile.categories]
        counts = Counter({key: 0 for key in REASON_NAMES})
        eligible = []
        for profile in candidates:
            reasons = _reasons(profile, order, event_date)
            counts.update(reasons)
            if not reasons:
                eligible.append(profile)
        eligible.sort(key=lambda profile: (profile.price_from_kzt, profile.id))
        if not candidates:
            status = "category_not_in_city"
            message = "В городе {} категории «{}» в каталоге нет.".format(order["city"], category)
        elif not eligible:
            status = "no_matches"
            message = "В городе {} есть {} кандидатов категории «{}», но ни один не прошёл условия.".format(
                order["city"], len(candidates), category)
        else:
            status = "matched"
            message = "Показано {} из {} подходящих профилей категории «{}» в городе {}.".format(
                min(len(eligible), 3), len(eligible), category, order["city"])
            if len(eligible) < 3:
                message += " Меньше трёх: в городе всего {} кандидатов, условия прошли {}.".format(
                    len(candidates), len(eligible))
        response = {
            "status": status,
            "message": message + _reason_summary(counts, order["event_date"]),
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "exclusion_counts": {key: counts[key] for key in REASON_NAMES},
            "cards": [_card(profile, category) for profile in eligible[:3]],
        }
        return self.explainer.enrich_response(order, response)
