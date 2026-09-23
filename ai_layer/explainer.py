"""Добавляет explanation, не меняя допуска, порядка и других полей ответа."""

from copy import deepcopy
from datetime import date

from .catalog import Catalog
from .config import DEFAULT_CSV, DEFAULT_INDEX, Settings
from .embeddings import OpenAIEmbeddings
from .evidence import EvidenceSelector

MONTHS = ("", "января", "февраля", "марта", "апреля", "мая", "июня",
          "июля", "августа", "сентября", "октября", "ноября", "декабря")


def _money(value):
    return format(int(value), ",").replace(",", " ")


def _same(left, right):
    return str(left).strip().casefold() == str(right).strip().casefold()


def _verify_eligible(order, row):
    """Защита от ошибочной интеграции: объяснять можно только прошедший профиль."""
    event_date = date.fromisoformat(order["event_date"])
    if not _same(order["city"], row["city"]):
        raise ValueError("Карточка не соответствует городу запроса")
    for field, requested in (("categories", order["category"]),
                             ("event_formats", order["event_type"])):
        if not any(_same(requested, value) for value in row[field].split("|")):
            raise ValueError("Карточка не соответствует " + field)
    if order["event_date"] in [value.strip() for value in row["busy_dates"].split("|")]:
        raise ValueError("Нельзя объяснять занятого подрядчика")
    if int(row["price_from_kzt"]) > int(order["budget_kzt"]):
        raise ValueError("Карточка превышает бюджет")
    if order.get("language") and not any(_same(order["language"], value)
                                         for value in row["languages"].split("|")):
        raise ValueError("Карточка не поддерживает язык")
    if order.get("duration_hours") is not None and row["max_hours"]:
        if float(order["duration_hours"]) > float(row["max_hours"]):
            raise ValueError("Карточка не покрывает длительность")
    return event_date


def _format_explanation(order, row, fragment, event_date):
    facts = ["На {} {} {} года профиль не отмечен занятым".format(
        event_date.day, MONTHS[event_date.month], event_date.year),
        "берёт формат «{}»".format(order["event_type"]),
        "цена от {} ₸ укладывается в бюджет {} ₸".format(
            _money(row["price_from_kzt"]), _money(order["budget_kzt"]))]
    if order.get("language"):
        facts.append("работает на языке «{}»".format(order["language"]))
    if order.get("duration_hours") is not None:
        if row["max_hours"]:
            facts.append("лимит {} ч покрывает запрошенные {:g} ч".format(
                row["max_hours"], float(order["duration_hours"])))
        else:
            facts.append("услуга не привязана к часам присутствия")
    result = "; ".join(facts) + "."
    if fragment is not None:
        evidence = " ".join(fragment.text.split()).strip(" .!?,;:")
        if evidence:
            result += " В описании: «{}».".format(evidence)
    return result


class AIExplainer:
    def __init__(self, catalog_csv=DEFAULT_CSV, index_path=DEFAULT_INDEX,
                 settings=None, embedder=None):
        self.catalog = Catalog(catalog_csv)
        self.settings = settings or Settings.load()
        if embedder is None and self.settings.openai_api_key:
            embedder = OpenAIEmbeddings(self.settings.openai_api_key, self.settings.model,
                                       self.settings.dimensions)
        self.selector = EvidenceSelector(self.catalog, embedder, self.settings.model,
                                         self.settings.dimensions, index_path)

    def prepare_index(self):
        return self.selector.prepare_index()

    def explain_card(self, order, card):
        row = self.catalog.profiles[card["id"]]
        event_date = _verify_eligible(order, row)
        if "price_from_kzt" in card and int(card["price_from_kzt"]) != int(row["price_from_kzt"]):
            raise ValueError("Цена карточки не соответствует CSV")
        fragment = self.selector.select(order, card["id"])
        return _format_explanation(order, row, fragment, event_date)

    def enrich_response(self, order, response):
        """Принимает результат backend по spec.md; возвращает копию с explanation."""
        enriched = deepcopy(response)
        cards = enriched.get("cards", [])
        if not cards:
            return enriched
        if enriched.get("status") != "matched" or len(cards) > 3:
            raise ValueError("AI-слой ожидает до трёх отфильтрованных карточек")
        for card in cards:
            card["explanation"] = self.explain_card(order, card)
        return enriched
