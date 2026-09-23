"""Ограниченное распознавание сокращений и опечаток без внешних библиотек."""

import re


SHORTCUTS = {
    "city": {"Алматы": ("ала", "алм", "алма", "алма-ата", "алма ата"),
             "Астана": ("аст", "нурсултан")},
    "category": {"Ведущий": ("вед",), "Фотограф": ("фот", "фото"),
                 "Видеограф": ("видеогр",), "Флорист": ("флор",),
                 "Банкетный зал": ("банкет",)},
    "event_type": {"свадьба": ("свад", "свадебный", "свадебную"),
                   "корпоратив": ("корп",), "конференция": ("конф",),
                   "день рождения": ("др", "д р"), "юбилей": ("юбил",)},
    "language": {"русский": ("рус",), "казахский": ("каз",), "английский": ("анг",)},
}

# Полные словоформы нужны для исправления ровно одной опечатки, например
# «свадбу» -> «свадьбу». Короткие сокращения нечётко не сравниваются.
WORD_FORMS = {
    "city": {"Алматы": ("алмате", "алмата", "алматы"), "Астана": ("астане", "астану")},
    "category": {"Ведущий": ("ведущий", "ведущего", "ведущая", "ведущую", "тамада", "тамаду"),
                 "Фотограф": ("фотографа", "фотографу", "фотосессия", "фотосессию"),
                 "Видеограф": ("видеографа",), "Флорист": ("флориста",),
                 "Декоратор": ("декоратора",), "Банкетный зал": ("банкетный", "банкетного")},
    "event_type": {"свадьба": ("свадьба", "свадьбу", "свадьбе", "свадьбы"),
                   "корпоратив": ("корпоратива", "корпоративе"),
                   "конференция": ("конференцию", "конференции"), "юбилей": ("юбилея",)},
    "language": {"русский": ("русском",), "казахский": ("казахском",),
                 "английский": ("английском",)},
}

MONTH_FORMS = (
    ("январь", "января", "янв"), ("февраль", "февраля", "фев"),
    ("март", "марта", "мар"), ("апрель", "апреля", "апр"),
    ("май", "мая", "мае"), ("июнь", "июня", "июн"),
    ("июль", "июля", "июл"), ("август", "августа", "авг"),
    ("сентябрь", "сентября", "сент", "сен"), ("октябрь", "октября", "окт"),
    ("ноябрь", "ноября", "ноя"), ("декабрь", "декабря", "дек"),
)


def normalize(text):
    return text.casefold().replace("ё", "е").replace("\u00a0", " ").replace("\u202f", " ")


def one_typo(left, right):
    """Одна вставка, потерянная/заменённая буква или перестановка соседних."""
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        mismatches = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        return (len(mismatches) <= 1 or
                len(mismatches) == 2 and mismatches[1] == mismatches[0] + 1 and
                left[mismatches[0]] == right[mismatches[1]] and
                left[mismatches[1]] == right[mismatches[0]])
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    for i, (a, b) in enumerate(zip(shorter, longer)):
        if a != b:
            return shorter[i:] == longer[i + 1:]
    return True


def _contains(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)", text)


def catalog_value(text, field, choices, aliases):
    # Точное название из выбранного CSV всегда приоритетнее сокращения/опечатки.
    for value in sorted(choices, key=lambda value: (-len(value), value)):
        if _contains(text, value):
            return value
    for value, stems in aliases.items():
        if value in choices and any(re.search(r"(?<!\w)" + re.escape(normalize(stem)), text)
                                    for stem in stems):
            return value
    for value, forms in SHORTCUTS.get(field, {}).items():
        if value in choices and any(_contains(text, form) for form in forms):
            return value

    tokens = set(re.findall(r"[^\W\d_]+", text))
    # Длинное начало одного слова можно дополнить, если оно однозначно.
    prefixes = {value for value in choices if " " not in value
                for token in tokens if len(token) >= 4 and normalize(value).startswith(token)}
    if prefixes:
        return next(iter(prefixes)) if len(prefixes) == 1 else None
    candidates = set()
    for value in choices:
        forms = [normalize(form) for form in (value,) + WORD_FORMS.get(field, {}).get(value, ())
                 if " " not in form]
        if any(len(token) >= 5 and len(form) >= 5 and token[0] == form[0] and one_typo(token, form)
               for token in tokens for form in forms):
            candidates.add(value)
    return next(iter(candidates)) if len(candidates) == 1 else None


def month_number(token):
    token = normalize(token)
    candidates = {number for number, forms in enumerate(MONTH_FORMS, 1)
                  if token in forms or len(token) >= 4 and any(form.startswith(token) for form in forms)}
    if not candidates and len(token) >= 5:
        candidates = {number for number, forms in enumerate(MONTH_FORMS, 1)
                      if any(len(form) >= 5 and token[0] == form[0] and one_typo(token, form) for form in forms)}
    return next(iter(candidates)) if len(candidates) == 1 else None
