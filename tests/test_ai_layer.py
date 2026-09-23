import json
import tempfile
import unittest
from pathlib import Path

from ai_layer import AIExplainer
from ai_layer.catalog import Catalog
from ai_layer.config import ROOT, Settings
from ai_layer.embeddings import EmbeddingError


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbeddingError("offline")
        return [[1.0, 0.0] if "свад" in text.casefold() else [0.0, 1.0] for text in texts]


class AILayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT / "examples/demo-cases.json").read_text(encoding="utf-8"))
        cls.explainer = AIExplainer(settings=Settings())

    def _response(self, case):
        cards = []
        for profile_id in case["expected"]["card_ids"]:
            row = self.explainer.catalog.profiles[profile_id]
            cards.append({"id": profile_id, "price_from_kzt": int(row["price_from_kzt"])})
        return {"status": case["expected"]["status"], "cards": cards,
                "candidate_count": case["expected"]["candidate_count"]}

    def test_fragments_have_exact_offsets(self):
        catalog = Catalog()
        self.assertEqual(len(catalog.profiles), 66)
        self.assertGreater(len(catalog.all_fragments()), 66)
        for fragment in catalog.all_fragments():
            description = catalog.profiles[fragment.profile_id]["description"]
            self.assertEqual(description[fragment.start:fragment.end], fragment.text)

    def test_all_demo_cards_are_grounded_and_preserve_order(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                original = self._response(case)
                result = self.explainer.enrich_response(case["request"], original)
                self.assertEqual(original["cards"][0].get("explanation") if original["cards"] else None, None)
                self.assertEqual([card["id"] for card in result["cards"]], case["expected"]["card_ids"])
                for card in result["cards"]:
                    self.assertIn(case["request"]["event_date"][-2:].lstrip("0") + " ", card["explanation"])
                    self.assertIn("₸", card["explanation"])
                    evidence = self.explainer.selector.select(case["request"], card["id"])
                    self.assertIsNotNone(evidence)
                    self.assertIn(" ".join(evidence.text.split()).strip(" .!?,;:"), card["explanation"])

    def test_dense_explanations_are_not_interchangeable(self):
        # Проверяем смысловую деталь отдельно от даты, цены и прочих общих фактов:
        # разные числа или имена сами по себе не выполняют критерий приёмки.
        details = {
            "HK-88430": ("корпоративных мероприятий", "деловых встреч"),
            "HK-29829": ("развлечения", "танцы"),
            "HK-27222": ("европейская подача", "тонкий юмор", "уважение к традициям"),
            "HK-44923": ("оригинальный сценарий",),
            "HK-44733": ("от 8 человек", "на 3000 человек"),
        }
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeEmbeddings()
            semantic = AIExplainer(settings=Settings(model="fake", dimensions=2), embedder=fake,
                                   index_path=Path(directory) / "index.json")
            semantic.prepare_index()
            for mode in ("local", "semantic", "offline"):
                explainer = self.explainer if mode == "local" else semantic
                fake.fail = mode == "offline"
                explainer.selector.query_cache.clear()
                for case in self.cases[:2]:
                    with self.subTest(mode=mode, case=case["id"]):
                        original = self._response(case)
                        result = explainer.enrich_response(case["request"], original)
                        cards = result["cards"]
                        self.assertEqual([card["id"] for card in cards], case["expected"]["card_ids"])
                        self.assertEqual(len(cards), 3)
                        repeated = explainer.enrich_response(case["request"], original)
                        self.assertEqual(result, repeated)

                        names = [explainer.catalog.profiles[card["id"]]["anon_name"] for card in cards]

                        def blinded(text):
                            text = " ".join(text.casefold().split())
                            for name in names:
                                text = text.replace(name.casefold(), "")
                            return " ".join(text.split())

                        sources = {card["id"]: blinded(explainer.catalog.profiles[card["id"]]["description"])
                                   for card in cards}
                        quotes = []
                        for card in cards:
                            text = card["explanation"]
                            self.assertIn(" В описании: «", text)
                            self.assertTrue(text.endswith("»."))
                            quote = blinded(text.split(" В описании: «", 1)[1][:-2])
                            self.assertTrue(quote)
                            for detail in details[card["id"]]:
                                self.assertIn(detail, quote)
                            # После удаления имён цитата должна подходить ровно
                            # одному исходному описанию из показанной тройки.
                            owners = [profile_id for profile_id, source in sources.items() if quote in source]
                            self.assertEqual(owners, [card["id"]])
                            quotes.append(quote)
                        self.assertEqual(len(set(quotes)), 3)

    def test_photographer_explanation_uses_experience_instead_of_farewell(self):
        order = dict(self.cases[0]["request"], category="Фотограф")
        row = self.explainer.catalog.profiles["HK-30583"]
        card = {"id": row["id"], "price_from_kzt": int(row["price_from_kzt"])}
        explanation = self.explainer.explain_card(order, card)
        self.assertIn("снимаю и концерты", explanation)
        self.assertNotIn("До встречи", explanation)
        fragment = self.explainer.selector.select(order, card["id"])
        self.assertEqual(row["description"][fragment.start:fragment.end], fragment.text)
        self.assertIn(" ".join(fragment.text.split()).strip(" .!?,;:"), explanation)

    def test_rejects_ineligible_card(self):
        case = self.cases[0]
        original = self._response(case)
        original["cards"][0]["id"] = "HK-44733"  # занят 10 октября
        with self.assertRaisesRegex(ValueError, "занятого"):
            self.explainer.enrich_response(case["request"], original)

    def test_semantic_similarity_cannot_promote_generic_praise(self):
        explainer = AIExplainer(settings=Settings(model="fake", dimensions=2), embedder=FakeEmbeddings())
        selector = explainer.selector
        selector.vectors = [[0.0, 1.0] if "ответственная" in item.text else [1.0, 0.0]
                            for item in selector.fragments]
        fragment = selector.select(self.cases[0]["request"], "HK-29829")
        self.assertIn("танцы", fragment.text)
        self.assertNotIn("ответственная", fragment.text)

    def test_optional_fields_and_null_hours(self):
        case = self.cases[2]
        order = dict(case["request"], language="русский", duration_hours=6)
        result = self.explainer.enrich_response(order, self._response(case))
        self.assertIn("русский", result["cards"][0]["explanation"])
        self.assertIn("не привязана к часам", result["cards"][0]["explanation"])

    def test_embedded_index_and_network_fallback(self):
        case = self.cases[2]
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeEmbeddings()
            explainer = AIExplainer(settings=Settings(model="fake", dimensions=2), embedder=fake,
                                    index_path=Path(directory) / "index.json")
            count = explainer.prepare_index()
            self.assertEqual(count, len(explainer.catalog.all_fragments()))
            before = fake.calls
            explainer.enrich_response(case["request"], self._response(case))
            self.assertEqual(fake.calls, before + 1)
            fake.fail = True
            explainer.selector.query_cache.clear()
            result = explainer.enrich_response(case["request"], self._response(case))
            self.assertTrue(result["cards"][0]["explanation"])


if __name__ == "__main__":
    unittest.main()
