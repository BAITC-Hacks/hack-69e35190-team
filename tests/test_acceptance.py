"""Критерии демонстрации: календарь, воспроизводимость и живой HTTP-ответ."""

import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from ai_layer import AIExplainer
from ai_layer.config import ROOT, Settings
from backend.api import make_server
from backend.catalog import Contractor
from backend.chat import ChatService
from backend.datasets import DatasetRegistry
from backend.service import RecommendationService


BASE = dict(city="Алматы", event_date="2026-10-10", event_type="корпоратив",
            category="Ведущий", budget_kzt=1500000)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.service = RecommendationService(explainer=AIExplainer(settings=Settings()))
        self.chat = ChatService(settings=Settings())

    def test_changed_date_names_the_actual_calendar_changes(self):
        response = self.chat.respond({"message": "11 октября", "slots": BASE}, self.service)
        comparison = response["result"]["date_comparison"]
        self.assertEqual(comparison["previous_date"], "2026-10-10")
        self.assertEqual(comparison["event_date"], "2026-10-11")
        self.assertEqual([(c["id"], c["change"], c["reason"]) for c in comparison["changes"]], [
            ("HK-88430", "removed", "busy"), ("HK-29829", "removed", "busy"),
            ("HK-44923", "added", "available"), ("HK-44733", "added", "available"),
        ])
        for change in comparison["changes"]:
            self.assertIn("2026-10-10", change["message"])
            self.assertIn("2026-10-11", change["message"])
        self.assertIn("занятости", comparison["message"])

    def test_comparison_does_not_blame_calendar_when_other_conditions_change(self):
        for message in ("11 октября, бюджет 100000", "10 октября", "бюджет 200000"):
            with self.subTest(message=message):
                response = self.chat.respond({"message": message, "slots": BASE}, self.service)
                self.assertNotIn("date_comparison", response["result"])

    def test_reload_and_csv_row_order_do_not_change_equal_price_ranking(self):
        # A becomes available and pushes D out of the top three even though D is free.
        # This must be explained as a ranking change, not invented calendar occupancy.
        rows = [dict(id=key, anon_name="Профиль " + key, categories="Ведущий", city="Алматы",
                     price_from_kzt="100000", event_formats="корпоратив", languages="русский",
                     max_hours="8", busy_dates="2026-10-10" if key == "A" else "",
                     description="Проводит корпоративные интерактивы по программе " + key + ".",
                     synthetic="true", city_imputed="false", price_imputed="false")
                for key in ("A", "B", "C", "D")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            results = []
            for ordered_rows in (rows, list(reversed(rows))):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(Contractor.__dataclass_fields__))
                    writer.writeheader()
                    writer.writerows(ordered_rows)
                service = RecommendationService(path, AIExplainer(path, settings=Settings()))
                results.append(service.recommend(BASE))
            self.assertEqual(results[0], results[1])
            self.assertEqual([c["id"] for c in results[0]["cards"]], ["B", "C", "D"])
            comparison = service.compare_dates(BASE, dict(BASE, event_date="2026-10-11"))
            self.assertEqual([(c["id"], c["reason"]) for c in comparison["changes"]],
                             [("D", "rank"), ("A", "available")])
            self.assertIn("На обе даты не отмечен занятым", comparison["changes"][0]["message"])

    def test_dense_rare_and_empty_http_scenarios_finish_with_explanations(self):
        cases = json.loads((ROOT / "examples/demo-cases.json").read_text(encoding="utf-8"))
        messages = [
            "Корпоратив в Алматы 10 октября, ведущий, бюджет 1500000",
            "Свадьба в Алматы 10 октября, флорист, бюджет 500000",
            "Корпоратив в Алматы 10 октября, ведущий, бюджет 1",
            "Свадьба в Зарубежье 10 октября, флорист, бюджет 500000",
        ]
        with tempfile.TemporaryDirectory() as directory:
            registry = DatasetRegistry(self.service, Path(directory), settings=Settings())
            server = make_server(port=0, registry=registry, chat=self.chat)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for message, expected in zip(messages, (cases[0], cases[2], cases[4], cases[3])):
                    with self.subTest(message=message):
                        request = Request("http://127.0.0.1:{}/chat".format(server.server_port),
                                          json.dumps({"message": message}).encode("utf-8"),
                                          headers={"Content-Type": "application/json"})
                        started = time.monotonic()
                        with urlopen(request, timeout=10) as response:
                            result = json.load(response)["result"]
                            self.assertEqual(response.status, 200)
                        self.assertLess(time.monotonic() - started, 10)
                        self.assertEqual(result["status"], expected["expected"]["status"])
                        self.assertEqual([c["id"] for c in result["cards"]], expected["expected"]["card_ids"])
                        self.assertTrue(result["message"])
                        for card in result["cards"]:
                            self.assertTrue(card["explanation"])
                metadata = registry.describe("default")
                self.assertEqual(metadata["category_counts"]["Ведущий"], 15)
                self.assertEqual(metadata["category_counts"]["Фотограф"], 12)
                self.assertEqual(metadata["category_counts"]["Банкетный зал"], 8)
                self.assertEqual(metadata["category_counts"]["Флорист"], 3)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
