"""Приёмочные регрессии: конкретные, целые цитаты и корректный кэш индекса."""

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from ai_layer import AIExplainer
from ai_layer.catalog import FRAGMENTATION_VERSION, fragments_for
from ai_layer.config import Settings
from ai_layer.embeddings import EmbeddingError
from ai_layer.evidence import GENERIC_PRAISE, INDEX_SCHEMA_VERSION, MAX_EVIDENCE_CHARS


class PraiseBiasedEmbeddings:
    """Заведомо отдаёт рекламным фразам максимальную близость запросу."""

    def __init__(self):
        self.fail = False

    def embed(self, texts):
        if self.fail:
            raise EmbeddingError("offline")
        return [[1.0, 0.0] if GENERIC_PRAISE.search(text.casefold()) or " для " in text
                else [0.0, 1.0] for text in texts]


class ExplanationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "index.json"
        self.settings = Settings(model="test", dimensions=2)
        self.fake = PraiseBiasedEmbeddings()
        self.local = AIExplainer(settings=Settings(), index_path=self.path)
        self.semantic = AIExplainer(settings=self.settings, embedder=self.fake,
                                    index_path=self.path)
        self.semantic.prepare_index()

    def modes(self):
        yield "local", self.local
        yield "semantic", self.semantic
        self.fake.fail = True
        self.semantic.selector.query_cache.clear()
        yield "offline", self.semantic

    def test_equal_price_bands_explain_different_instrumentation(self):
        order = {"city": "Алматы", "event_date": "2026-10-01", "event_type": "свадьба",
                 "category": "Лайв-бэнд", "budget_kzt": 1150000}
        cards = [{"id": "HK-23752"}, {"id": "HK-83709"}]
        for mode, explainer in self.modes():
            with self.subTest(mode=mode):
                response = {"status": "matched", "cards": cards}
                result = explainer.enrich_response(order, response)
                self.assertEqual([card["id"] for card in result["cards"]],
                                 [card["id"] for card in cards])
                quotes = []
                for card in result["cards"]:
                    explanation = card["explanation"]
                    self.assertNotIn("идеально впишется", explanation)
                    quote = explanation.split(" В описании: «", 1)[1][:-2]
                    row = explainer.catalog.profiles[card["id"]]
                    self.assertIn(quote, " ".join(row["description"].split()))
                    # Ни вымышленное имя, ни сохранившееся название группы
                    # не должны быть единственным различием объяснений.
                    for name in (row["anon_name"], "Thunder Breath Band", "Eva Sound"):
                        quote = quote.replace(name, "")
                    quotes.append(quote)
                self.assertIn("два вокалиста", quotes[0])
                self.assertIn("тромбон", quotes[0])
                self.assertIn("4 вокалиста", quotes[1])
                self.assertIn("струнный квартет", quotes[1])
                self.assertNotEqual(quotes[0], quotes[1])
                self.assertEqual(result, explainer.enrich_response(order, response))

    def test_long_lists_do_not_become_broken_card_explanations(self):
        cases = (
            ("HK-31819", "2026-09-23", "Лайв-бэнд", ("Большой состав", "Расширенный состав"), "тромбон"),
            ("HK-44733", "2026-10-11", "Ведущий", ("Веду как",), "на 3000 человек"),
        )
        for mode, explainer in self.modes():
            for profile_id, day, category, beginning, detail in cases:
                with self.subTest(mode=mode, profile=profile_id):
                    order = {"city": "Алматы", "event_date": day, "event_type": "корпоратив",
                             "category": category, "budget_kzt": 1500000}
                    explanation = explainer.explain_card(order, {"id": profile_id})
                    self.assertTrue(any("В описании: «" + value in explanation for value in beginning))
                    self.assertIn(detail, explanation)
                    self.assertNotIn("«музыканта)", explanation)
                    self.assertNotIn("(Наруто»", explanation)
                    for fragment in explainer.catalog.fragments[profile_id]:
                        self.assertEqual(fragment.text.count("("), fragment.text.count(")"))
                        source = explainer.catalog.profiles[profile_id]["description"]
                        self.assertEqual(source[fragment.start:fragment.end], fragment.text)

    def test_unstructured_long_sentence_remains_whole(self):
        description = "Оркестр выступает с " + "профессиональными музыкантами " * 15 + "(12 человек)."
        fragments = fragments_for("test", description)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0].text, description)
        self.assertEqual(description[fragments[0].start:fragments[0].end], description)

    def test_missing_space_between_sentences_is_not_a_joined_quote(self):
        text = "Выступаем в камерных залах.Команда играет джазовые композиции."
        fragments = fragments_for("test", text)
        self.assertEqual([fragment.text for fragment in fragments], [
            "Выступаем в камерных залах.", "Команда играет джазовые композиции.",
        ])
        initials = "Музыкант И.Иванов выступает с оркестром на ул.Абая."
        self.assertEqual([f.text for f in fragments_for("test", initials)], [initials])

    def _order(self, row):
        busy = set(row["busy_dates"].split("|"))
        free = next((date(2026, 9, 23) + timedelta(days=offset)).isoformat()
                    for offset in range(100)
                    if (date(2026, 9, 23) + timedelta(days=offset)).isoformat() not in busy)
        return {"city": row["city"], "category": row["categories"].split("|")[0],
                "event_type": row["event_formats"].split("|")[0], "event_date": free,
                "budget_kzt": int(row["price_from_kzt"])}

    def test_additional_catalog_advertising_uses_facts_in_all_modes(self):
        for mode, explainer in self.modes():
            for profile_id in ("HK-19103", "HK-25279", "HK-57480"):
                with self.subTest(mode=mode, profile=profile_id):
                    row = explainer.catalog.profiles[profile_id]
                    order = self._order(row)
                    explanation = explainer.explain_card(order, {"id": profile_id})
                    self.assertNotIn("свяжитесь с нами", explanation)
                    self.assertNotIn("сверкаем", explanation)
                    self.assertNotIn("атмосферу.Команда", explanation)
                    self.assertEqual(explanation.count("."), 2)
                    if profile_id == "HK-19103":
                        self.assertIn("команда джигитов", explanation)
                    elif profile_id == "HK-25279":
                        self.assertIsNone(explainer.selector.select(order, profile_id))
                        self.assertNotIn(" В описании: «", explanation)
                        self.assertIn("Краткой конкретной цитаты в описании нет", explanation)
                        self.assertIn("языки — русский", explanation)
                        self.assertIn("максимум присутствия — 5 ч", explanation)
                    else:
                        self.assertTrue("хореографией" in explanation or "хитов 90-х" in explanation)
                        fragment = explainer.selector.select(order, profile_id)
                        self.assertEqual(row["description"][fragment.start:fragment.end], fragment.text)

    def test_unbounded_description_falls_back_without_cutting_a_quote(self):
        profile_id = "HK-25279"
        row = self.local.catalog.profiles[profile_id]
        row["description"] = "Исполняем музыку " + "с живым оркестром " * 1000
        self.local.catalog.fragments[profile_id] = fragments_for(profile_id, row["description"])
        self.assertGreater(len(self.local.catalog.fragments[profile_id][0].text), MAX_EVIDENCE_CHARS)
        explanation = self.local.explain_card(self._order(row), {"id": profile_id})
        self.assertNotIn(" В описании: «", explanation)
        self.assertIn("Краткой конкретной цитаты в описании нет", explanation)
        self.assertLess(len(explanation), 600)

    def test_generic_only_description_does_not_become_an_explanation(self):
        selector = self.local.selector
        selector.catalog.profiles["generic"] = {"anon_name": "Тест"}
        selector.catalog.fragments["generic"] = fragments_for(
            "generic", "Отличный выбор для мероприятия! Идеально впишется в любой формат мероприятия."
        )
        order = {"city": "Алматы", "event_type": "свадьба", "category": "Лайв-бэнд"}
        self.assertIsNone(selector.select(order, "generic"))

    def test_old_or_different_segmentation_index_is_never_reused(self):
        original = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(original["schema"], INDEX_SCHEMA_VERSION)
        self.assertEqual(original["fragmentation_version"], FRAGMENTATION_VERSION)
        for change in ({"schema": 1}, {"fragmentation_version": FRAGMENTATION_VERSION - 1}):
            with self.subTest(change=change):
                # Число векторов и digest специально совпадают: этого недостаточно.
                self.path.write_text(json.dumps(dict(original, **change)), encoding="utf-8")
                reloaded = AIExplainer(settings=self.settings, embedder=self.fake,
                                       index_path=self.path)
                self.assertIsNone(reloaded.selector.vectors)
        self.path.write_text(json.dumps(original), encoding="utf-8")
        reloaded = AIExplainer(settings=self.settings, embedder=self.fake, index_path=self.path)
        self.assertIsNotNone(reloaded.selector.vectors)


if __name__ == "__main__":
    unittest.main()
