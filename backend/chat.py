"""Разбор диалога на сервере: локальные правила, опциональный LLM, валидация кодом."""

import json
import re
import threading
from collections import OrderedDict

from ai_layer.config import Settings
from ai_layer.intent import FIELDS, IntentExtractor

REQUIRED = ("city", "category", "event_type", "event_date", "budget_kzt")
ALIASES = {
    "city": {"Алматы": ("алмат",), "Астана": ("астан", "нур-султан"), "Зарубежье": ("зарубеж", "за рубежом")},
    "category": {
        "Ведущий церемонии": ("церемони", "регистраци"), "Фото и видеобудки": ("фотобуд", "видеобуд"),
        "Национальный ансамбль": ("ансамбл",), "Танцевальный коллектив": ("танцевальн", "танцор"),
        "Загородная площадка": ("загород",), "Банкетный зал": ("банкетн", "зал"),
        "Подарки и сувениры": ("подарк", "сувенир"), "Лайв-бэнд": ("бэнд", "живая музык", "живую музык"),
        "Инструменталист": ("инструментал", "скрипач", "саксофон"), "Шоу-программа": ("шоу",),
        "Декоратор": ("декор", "оформлени"), "Флорист": ("флорист", "цветы", "цветов"),
        "Видеограф": ("видеограф", "видеосъем", "видеосъём"),
        "Фотограф": ("фотограф", "фотосесси", "фотосъем", "фотосъём", "фотографирован"),
        "Ресторан": ("ресторан",), "Отель": ("отел", "гостиниц"), "Ведущий": ("ведущ", "тамад"),
    },
    "event_type": {"свадьба": ("свадьб",), "той": ("той",), "корпоратив": ("корпоратив",),
                   "конференция": ("конференци",), "юбилей": ("юбиле",), "день рождения": ("день рождения", "днюх")},
    "language": {"русский": ("русск",), "казахский": ("казахск",), "английский": ("англ",)},
}
MONTHS = ("январ", "феврал", "март", "апрел", "ма[яе]", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр")


def vocabulary(service):
    return {"city": list(service.cities), "category": sorted(service.categories.values()),
            "event_type": list(service.event_types), "language": list(service.languages),
            "calendar_start": str(service.calendar_start), "calendar_end": str(service.calendar_end)}


def local_updates(text, previous, vocab):
    t = text.casefold().replace("\u00a0", " ")
    updates = {}
    for field, aliases in ALIASES.items():
        for value in sorted(vocab[field], key=lambda value: (-len(value), value)):
            if re.search(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", t):
                updates[field] = value
                break
        if field not in updates:
            for value, stems in aliases.items():
                if any(re.search(r"(?<!\w)" + re.escape(stem), t) for stem in stems):
                    updates[field] = value
                    break
    if re.search(r"(?:язык|языке)\s*(?:не\s*важ[а-я]*|любой|без\s*разницы)", t):
        updates["language"] = None
    duration = re.search(r"(?<![\d:])(\d+(?:[.,]\d+)?)\s*(?:час(?:а|ов)?|ч)(?!\w)", t)
    if duration:
        updates["duration_hours"] = float(duration[1].replace(",", "."))
        t = t[:duration.start()] + " " + t[duration.end():]
    # Удаление времени не позволяет принять его за дату или бюджет.
    t = re.sub(r"\b\d{1,2}:\d{2}\b", " ", t)
    year = int(vocab["calendar_start"][:4])
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    numeric = re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{4}))?\b", t)
    if iso:
        updates["event_date"] = iso[0]
        t = t.replace(iso[0], " ", 1)
    elif numeric:
        updates["event_date"] = "{:04d}-{:02d}-{:02d}".format(int(numeric[3] or year), int(numeric[2]), int(numeric[1]))
        t = t.replace(numeric[0], " ", 1)
    else:
        for month, pattern in enumerate(MONTHS, 1):
            match = re.search(r"\b(\d{1,2})\s+(?:" + pattern + r")[а-я]*(?:\s+(\d{4}))?", t)
            if match:
                updates["event_date"] = "{:04d}-{:02d}-{:02d}".format(int(match[2] or year), month, int(match[1]))
                t = t.replace(match[0], " ", 1)
                break
    budget = re.search(r"(?:бюджет(?:ом)?|до|за)\s*[:=]?\s*(\d+(?:[ .]\d{3})*(?:[.,]\d+)?)\s*(млн|миллион[а-я]*|тыс[а-я.]*|[кk](?!\w))?", t)
    if not budget:
        budget = re.search(r"\b(\d+(?:[ .]\d{3})*(?:[.,]\d+)?)\s*(млн|тыс[а-я.]*|[кk](?!\w)|тенге|₸)", t)
    missing = next((key for key in REQUIRED if key not in (previous | updates)), None)
    if not budget and missing == "budget_kzt":
        budget = re.fullmatch(r"\s*(\d+(?:[ ]\d{3})*)\s*()", t)
    if budget:
        amount = float(budget[1].replace(" ", "").replace(",", "."))
        unit = budget[2] or ""
        amount *= 1_000_000 if unit.startswith(("млн", "миллион")) else 1000 if unit.startswith(("тыс", "к", "k")) else 1
        if not amount.is_integer():
            raise ValueError("Бюджет должен быть целым числом тенге")
        updates["budget_kzt"] = int(amount)
    return updates


def validate_partial(service, slots):
    if not isinstance(slots, dict) or slots.keys() - set(FIELDS):
        raise ValueError("Некорректные поля slots")
    base = {"city": service.cities[0], "category": "Категория", "event_type": service.event_types[0],
            "event_date": str(service.calendar_start), "budget_kzt": 1}
    # Валидируем даже неполный запрос; неизвестные даты/языки не обходят проверки.
    cleaned = service.validate(base | slots)
    return {key: cleaned.get(key) for key in slots}


class ChatService:
    def __init__(self, settings=None, extractor=None):
        self.settings = settings or Settings.load()
        self.extractor = extractor or (IntentExtractor(self.settings) if self.settings.openai_api_key else None)
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def respond(self, data, service):
        text = data.get("message")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000:
            raise ValueError("Сообщение должно содержать от 1 до 4000 символов")
        previous = validate_partial(service, data.get("slots", {}))
        vocab = vocabulary(service)
        updates = local_updates(text, previous, vocab)
        warnings, mode = [], "local"
        merged = previous | updates
        understood = bool(updates)
        if self.extractor and (not updates or any(not merged.get(key) for key in REQUIRED)):
            cache_key = json.dumps([service.explainer.catalog.digest, vocab, text, previous], ensure_ascii=False, sort_keys=True)
            try:
                with self.lock:
                    extra = self.cache.get(cache_key)
                if extra is None:
                    extra = self.extractor.extract(text, previous, vocab)
                    extra = validate_partial(service, extra)
                    with self.lock:
                        self.cache[cache_key] = extra
                        if len(self.cache) > 256:
                            self.cache.popitem(last=False)
                merged = previous | extra | updates  # Явные локальные значения приоритетнее LLM.
                understood = understood or bool(extra)
                mode = "openai"
            except Exception:
                warnings.append("AI-разбор недоступен: использованы локальные правила; проверьте распознанные поля")
        slots = validate_partial(service, merged)
        missing = next((key for key in REQUIRED if not slots.get(key)), None)
        prompts = {"city": "В каком городе ищем подрядчика?", "category": "Кого ищем — какая категория подрядчика?",
                   "event_type": "Какой тип мероприятия?", "event_date": "Укажите дату с {} по {}".format(service.calendar_start, service.calendar_end),
                   "budget_kzt": "Какой бюджет в тенге?"}
        if re.search(r"\b\d{1,2}:\d{2}\b", text):
            warnings.append("Календарь учитывает весь день; время начала отдельно не проверяется")
        if not understood and missing is None:
            return {"slots": slots, "missing_field": "clarification",
                    "prompt": "Не удалось распознать изменение. Укажите конкретно, например: «бюджет 500000» или «11 октября»",
                    "choices": [], "parser": mode, "warnings": warnings, "result": None}
        return {"slots": slots, "missing_field": missing, "prompt": prompts.get(missing),
                "choices": vocab.get(missing, []), "parser": mode, "warnings": warnings,
                "result": None if missing else service.recommend({k: v for k, v in slots.items() if v is not None})}
