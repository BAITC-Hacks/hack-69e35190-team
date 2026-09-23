"""Разбор диалога на сервере: локальные правила, опциональный LLM, валидация кодом."""

import json
import re
import threading
from collections import OrderedDict
from decimal import Decimal

from ai_layer.config import Settings
from ai_layer.intent import FIELDS, IntentExtractor
from .text_matching import catalog_value, month_number, normalize

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
AMOUNT = r"[+-]?\d+(?: \d{3})*(?:[.,]\d+)?"
MONEY_UNIT = r"(?:млн|миллион(?:а|ов)?|тыс(?:яч(?:а|и|у)?)?\.?|[кk]|тенге|тг|₸)(?!\w)"


def _budget_match(text, unit_required=False):
    # Сначала сумма с единицей: «1.5 млн» должна уйти из текста ДО поиска дат.
    with_unit = re.search(r"(?<![\w:])(" + AMOUNT + r")\s*(" + MONEY_UNIT + r")", text)
    if with_unit or unit_required:
        return with_unit
    return re.search(r"(?<!\w)(?:бюджет(?:ом)?|бюдж|до|за)\s*[:=]?\s*(" + AMOUNT +
                     r")\s*(" + MONEY_UNIT + r")?", text)


def _budget_amount(match):
    amount = Decimal(match[1].replace(" ", "").replace(",", "."))
    unit = match[2] or ""
    amount *= 1_000_000 if unit.startswith(("млн", "миллион")) else 1000 if unit.startswith(("тыс", "к", "k")) else 1
    if amount != amount.to_integral_value():
        raise ValueError("Бюджет должен быть целым числом тенге")
    return int(amount)


def vocabulary(service):
    return {"city": list(service.cities), "category": sorted(service.categories.values()),
            "event_type": list(service.event_types), "language": list(service.languages),
            "calendar_start": str(service.calendar_start), "calendar_end": str(service.calendar_end)}


def local_updates(text, previous, vocab):
    t = normalize(text)
    updates = {}
    for field, aliases in ALIASES.items():
        value = catalog_value(t, field, vocab[field], aliases)
        if value is not None:
            updates[field] = value
    if re.search(r"(?:язык|языке)\s*(?:не\s*важ[а-я]*|любой|без\s*разницы)", t):
        updates["language"] = None
    duration = re.search(r"(?<![\d:])(\d+(?:[.,]\d+)?)\s*(?:час(?:а|ов)?|ч)(?!\w)", t)
    if duration:
        updates["duration_hours"] = float(duration[1].replace(",", "."))
        t = t[:duration.start()] + " " + t[duration.end():]
    # Удаление времени не позволяет принять его за дату или бюджет.
    t = re.sub(r"\b\d{1,2}:\d{2}\b", " ", t)
    budget = _budget_match(t, unit_required=True)
    if budget:
        updates["budget_kzt"] = _budget_amount(budget)
        t = t[:budget.start()] + " " + t[budget.end():]
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
        for match in re.finditer(r"\b(\d{1,2})(?:-?(?:го|ое|е))?\s*([а-я]{3,})\.?(?:\s+(\d{4})(?!\d))?(?!\w)", t):
            month = month_number(match[2])
            if month is not None:
                updates["event_date"] = "{:04d}-{:02d}-{:02d}".format(int(match[3] or year), month, int(match[1]))
                t = t.replace(match[0], " ", 1)
                break
    if budget is None:
        budget = _budget_match(t)
        if budget:
            updates["budget_kzt"] = _budget_amount(budget)
    missing = next((key for key in REQUIRED if key not in (previous | updates)), None)
    if not budget and missing == "budget_kzt":
        budget = re.fullmatch(r"\s*(" + AMOUNT + r")\s*()", t)
        if budget:
            updates["budget_kzt"] = _budget_amount(budget)
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
        result = None
        if missing is None:
            order = {key: value for key, value in slots.items() if value is not None}
            result = service.recommend(order)
            if all(previous.get(key) for key in REQUIRED):
                comparison = service.compare_dates(
                    {key: value for key, value in previous.items() if value is not None}, order)
                if comparison is not None:
                    result["date_comparison"] = comparison
        return {"slots": slots, "missing_field": missing, "prompt": prompts.get(missing),
                "choices": vocab.get(missing, []), "parser": mode, "warnings": warnings,
                "result": result}
