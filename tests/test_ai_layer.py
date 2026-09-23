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
        case = self.cases[0]
        result = self.explainer.enrich_response(case["request"], self._response(case))
        explanations = [card["explanation"] for card in result["cards"]]
        self.assertEqual(len(set(explanations)), 3)
        self.assertTrue(any("корпоратив" in text.lower() and "делов" in text.lower() for text in explanations))
        self.assertTrue(any("танц" in text.lower() for text in explanations))
        self.assertTrue(any("традиц" in text.lower() or "двуязыч" in text.lower() for text in explanations))

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
