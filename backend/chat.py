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
HOURS = r"(?:час(?:а|ов)?|ч\.?)(?![^\W\d])"
MINUTES = r"(?:минут(?:а|ы)?|мин\.?)(?!\w)"
NUMBER = r"[+-]?\d+(?:[.,]\d+)?"
CLOCK = (r"(?<!\w)в\s+\d{1,2}\s*" + HOURS +
         r"(?:\s*(?:и\s*)?\d{1,2}\s*" + MINUTES + r")?|\b\d{1,2}:\d{2}\b")
CLEAR_LANGUAGE = r"(?:язык|языке)\s*(?:не\s*важ[а-я]*|любой|без\s*разницы)"
CLEAR_DURATION = (r"(?:длительность|продолжительность|часы)\s*"
                  r"(?:не\s*важ[а-я]*|люб[а-я]*|без\s*(?:разницы|ограничений))"
                  r"|без\s+ограничений\s+(?:по\s+)?(?:длительности|времени)")


def affirmative_text(text):
    """Отброшенное значение в «не X, а Y» не участвует в извлечении.

    Верхние границы и отмена необязательных условий — не отрицание значения.
    Другие отрицания ниже требуют уточнения, а не выбора отвергнутого варианта.
    """
    return re.sub(r"(?<!\w)не\s+(?!(?:важ[а-я]*|более|меньше|позже|раньше|дороже|выше)\b)"
                  r"[^;!?\n]*?\s*,?\s+а\s+", "", normalize(text))


def _budget_match(text, unit_required=False):
    # Сначала сумма с единицей: «1.5 млн» должна уйти из текста ДО поиска дат.
    with_unit = re.search(r"(?<![\w:])(" + AMOUNT + r")\s*(" + MONEY_UNIT + r")", text)
    if with_unit or unit_required:
        return with_unit
    limit = r"(?:не\s+(?:более|дороже|выше)|максимум|до|в\s+пределах)"
    return re.search(r"(?<!\w)(?:бюджет(?:ом)?|бюдж|" + limit + r"|за)\s*[:=]?\s*"
                     r"(?:" + limit + r"\s*)?(" + AMOUNT +
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
    t = affirmative_text(text)
    updates = {}
    for field, aliases in ALIASES.items():
        value = catalog_value(t, field, vocab[field], aliases)
        if value is not None:
            updates[field] = value
    if re.search(CLEAR_LANGUAGE, t):
        updates["language"] = None
    if re.search(CLEAR_DURATION, t):
        updates["duration_hours"] = None
    # «В 15 часов» — начало мероприятия, «на 3 часа» — длительность.
    t = re.sub(CLOCK, " ", t)
    duration = re.search(r"(?<![\d:])(" + NUMBER + r")\s*" + HOURS +
                         r"(?:\s*(?:и\s*)?(\d+(?:[.,]\d+)?)\s*" + MINUTES + r")?", t)
    if duration:
        hours = Decimal(duration[1].replace(",", "."))
        minutes = Decimal((duration[2] or "0").replace(",", "."))
        updates["duration_hours"] = float(hours + minutes / 60)
        t = t[:duration.start()] + " " + t[duration.end():]
    minutes = re.search(r"(?<![\d:])(" + NUMBER + r")\s*" + MINUTES, t)
    if minutes:
        if duration:
            raise ValueError("Укажите одну длительность, например: «на 6 часов 30 минут»")
        updates["duration_hours"] = float(Decimal(minutes[1].replace(",", ".")) / 60)
        t = t[:minutes.start()] + " " + t[minutes.end():]
    if re.search(r"(?<![\d:])" + NUMBER + r"\s*" + HOURS, t):
        raise ValueError("Укажите одну длительность мероприятия")
    budget = _budget_match(t, unit_required=True)
    if budget:
        updates["budget_kzt"] = _budget_amount(budget)
        t = t[:budget.start()] + " " + t[budget.end():]
    year = int(vocab["calendar_start"][:4])
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    numeric = re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{4}|\d{2}))?\b(?![./\d])", t)
    def explicit_year(value):
        return int(value) + (2000 if len(value) == 2 else 0) if value else year
    if iso:
        updates["event_date"] = iso[0]
        t = t.replace(iso[0], " ", 1)
    elif numeric:
        updates["event_date"] = "{:04d}-{:02d}-{:02d}".format(explicit_year(numeric[3]), int(numeric[2]), int(numeric[1]))
        t = t.replace(numeric[0], " ", 1)
    else:
        for match in re.finditer(r"\b(\d{1,2})(?:-?(?:го|ое|е))?\s*([а-я]{3,})\.?"
                                 r"(?:\s+(\d{4}|\d{2})(?!\d)(?:\s*г(?:ода|од|\.)?)?)?(?!\w)", t):
            month = month_number(match[2])
            if month is not None:
                updates["event_date"] = "{:04d}-{:02d}-{:02d}".format(explicit_year(match[3]), month, int(match[1]))
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


def unresolved_constraints(text, updates, vocab):
    """Не запускаем подбор со старым значением явно названного нового условия."""
    t = affirmative_text(text)
    unresolved = set()
    mentions = {
        "budget_kzt": r"(?<!\w)бюдж[а-я]*|(?<![\w:])" + AMOUNT + r"\s*" + MONEY_UNIT,
        "duration_hours": (r"(?:длительност|продолжительност)[а-я]*|(?<!\w)(?:" + HOURS + r"|" + MINUTES +
                           r")|(?<![\d:])" + NUMBER + r"\s*(?:" + HOURS + r"|" + MINUTES + r")"),
        "language": r"(?<!\w)язык[а-я]*",
        "city": r"(?<!\w)город[а-я]*",
        "category": r"(?<!\w)категори[а-я]*",
        "event_date": r"(?<!\w)дат(?:а|у|е|ы|ой)(?!\w)|\b\d{1,2}[./]\d{1,2}",
    }
    without_clock = re.sub(CLOCK, " ", t)
    # Десятичная сумма/длительность — не новая дата. Используем тот же порядок
    # маскирования, что и при извлечении, иначе «6.5 часов» теряет прежнюю дату.
    date_text = re.sub(r"(?<![\w:])" + AMOUNT + r"\s*" + MONEY_UNIT, " ", without_clock)
    date_text = re.sub(r"(?<![\d:])" + NUMBER + r"\s*(?:" + HOURS + r"|" + MINUTES + r")", " ", date_text)
    for field, pattern in mentions.items():
        if re.search(pattern, date_text if field == "event_date" else without_clock) and field not in updates:
            unresolved.add(field)
    # Именованные языки вне словаря нельзя превращать в «язык не важен».
    # Словоформы импортированных языков распознаются через catalog_value.
    for match in re.finditer(r"(?<!\w)(?:на\s+|по[- ])([а-я]+(?:ском|цком|ски|цки)|хинди|иврите|урду|фарси)(?!\w)", t):
        if catalog_value(match[1], "language", vocab["language"], ALIASES["language"]) is None:
            unresolved.add("language")
    for match in re.finditer(r"(?<!\w)язык[а-я]*\s*[:=—-]?\s+([а-я]+(?:\s+(?:и|или)\s+[а-я]+)*)", t):
        if re.match(CLEAR_LANGUAGE, t[match.start():]):
            continue
        tokens = re.findall(r"[а-я]+", match[1])
        # Проверяем каждое явно перечисленное название, а не только первое.
        named = [token for token in tokens if token.endswith(("ий", "ом")) or
                 token in ("хинди", "иврит", "урду", "фарси")]
        values = {catalog_value(token, "language", vocab["language"], ALIASES["language"]) for token in named}
        if None in values or len(values) > 1:
            unresolved.add("language")
    # «Не фотограф» не означает «Фотограф». Без явной замены спросим пользователя.
    negatives = re.finditer(r"(?<!\w)не\s+(?!(?:важ[а-я]*|более|меньше|позже|раньше|дороже|выше)\b)"
                           r"([^,;.!?]+)", t)
    for negative in negatives:
        for field, aliases in ALIASES.items():
            if catalog_value(negative[1], field, vocab[field], aliases) is not None:
                unresolved.add(field)
        if re.search(r"\d", negative[1]):
            unresolved.update(key for key in updates if key in ("event_date", "budget_kzt", "duration_hours"))
    return unresolved


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
        raw_pending = data.get("pending_fields", [])
        if (not isinstance(raw_pending, list) or
                any(not isinstance(field, str) or field not in FIELDS for field in raw_pending)):
            raise ValueError("pending_fields: требуется список известных полей заказа")
        pending = set(raw_pending)
        vocab = vocabulary(service)
        updates = local_updates(text, previous, vocab)
        newly_unresolved = unresolved_constraints(text, updates, vocab)
        unresolved = (pending - updates.keys()) | newly_unresolved
        warnings, mode = [], "local"
        merged = previous | updates
        understood = bool(updates)
        if self.extractor and (unresolved or not updates or any(not merged.get(key) for key in REQUIRED)):
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
                # Перенесённое спорное поле нельзя снять ответом модели на сообщение
                # о другом условии: нужен локальный разбор либо явное упоминание поля.
                confirmed = updates.keys() | (extra.keys() & newly_unresolved)
                unresolved = (pending - confirmed) | unresolved_constraints(text, extra | updates, vocab)
                understood = understood or bool(extra)
                mode = "openai"
            except Exception:
                warnings.append("AI-разбор недоступен: использованы локальные правила; проверьте распознанные поля")
        if re.search(CLOCK, normalize(text)):
            warnings.append("Календарь учитывает весь день; время начала отдельно не проверяется")
        if unresolved:
            # Не сохраняем прежнее значение спорного условия как будто оно подтверждено.
            # Остальные новые поля сохраняются, поэтому их не придётся вводить повторно.
            slots = validate_partial(service, {key: value for key, value in merged.items() if key not in unresolved})
            labels = {"city": "город", "category": "категорию", "event_type": "формат мероприятия",
                      "event_date": "дату", "budget_kzt": "бюджет в тенге",
                      "language": "язык из каталога или «язык не важен»",
                      "duration_hours": "длительность, например «6 часов 30 минут», или «длительность не важна»"}
            fields = sorted(unresolved)
            return {"slots": slots, "missing_field": "clarification",
                    "prompt": "Не удалось однозначно распознать условие. Уточните " +
                              "; ".join(labels[field] for field in fields) + ".",
                    "choices": vocab.get(fields[0], []), "parser": mode, "warnings": warnings,
                    "pending_fields": fields, "result": None}
        slots = validate_partial(service, merged)
        missing = next((key for key in REQUIRED if not slots.get(key)), None)
        prompts = {"city": "В каком городе ищем подрядчика?", "category": "Кого ищем — какая категория подрядчика?",
                   "event_type": "Какой тип мероприятия?", "event_date": "Укажите дату с {} по {}".format(service.calendar_start, service.calendar_end),
                   "budget_kzt": "Какой бюджет в тенге?"}
        if not understood and missing is None:
            return {"slots": slots, "missing_field": "clarification",
                    "prompt": "Не удалось распознать изменение. Укажите конкретно, например: «бюджет 500000» или «11 октября»",
                    "choices": [], "parser": mode, "warnings": warnings, "pending_fields": [], "result": None}
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
                "pending_fields": [], "result": result}
