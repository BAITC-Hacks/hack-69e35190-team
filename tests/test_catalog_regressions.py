"""Регрессии аудита: регистр CSV не меняет допуск; слишком большие числа — HTTP 400."""

import csv
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_layer import AIExplainer
from ai_layer.config import Settings
from backend.api import make_server
from backend.catalog import Contractor
from backend.chat import ChatService
from backend.datasets import DatasetRegistry
from backend.service import RecommendationService, validate_order

ORDER = dict(city="Алматы", event_date="2026-10-10", event_type="корпоратив",
             category="Ведущий", budget_kzt=100000, duration_hours=6, language="русский")


def profiles_csv(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(Contractor.__dataclass_fields__))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def profile(profile_id, **changes):
    return dict(id=profile_id, anon_name="Тест " + profile_id, categories="Ведущий", city="Алматы",
                price_from_kzt="100000", event_formats="корпоратив", languages="русский",
                max_hours="6", busy_dates="", description="Авторская программа для деловых событий " + profile_id,
                synthetic="true", city_imputed="false", price_imputed="false") | changes


class CatalogRegressionTests(unittest.TestCase):
    def test_case_variants_pass_all_filters_and_row_order_is_irrelevant(self):
        variants = ({"city": "алматы"}, {"categories": "ведущий"},
                    {"event_formats": "Корпоратив"}, {"languages": "Русский"},
                    {"city": " АЛМАТЫ ", "categories": "ведущий|Ведущий", "languages": " РУССКИЙ ",
                     "event_formats": "КОРПОРАТИВ"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            for variant in variants:
                with self.subTest(variant=variant):
                    responses = []
                    for rows in ([profile("A"), profile("B", **variant)],
                                 [profile("B", **variant), profile("A")]):
                        content = profiles_csv(rows)
                        path.write_text(content, encoding="utf-8")
                        service = RecommendationService(path, AIExplainer(path, settings=Settings()))
                        result = service.recommend(ORDER)
                        self.assertEqual(result["candidate_count"], 2)
                        self.assertEqual(result["eligible_count"], 2)
                        self.assertEqual([card["id"] for card in result["cards"]], ["A", "B"])
                        self.assertTrue(all(count == 0 for count in result["exclusion_counts"].values()))
                        self.assertTrue(all(card["city"] == "Алматы" for card in result["cards"]))
                        self.assertEqual(path.read_text(encoding="utf-8"), content.replace("\r\n", "\n"))
                        responses.append(result)
                    self.assertEqual(responses[0], responses[1])

    def test_imported_case_variants_survive_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = DatasetRegistry(directory=Path(directory), settings=Settings())
            try:
                content = profiles_csv([profile("A"), profile("B", city="алматы", categories="ведущий",
                                                                event_formats="Корпоратив", languages="Русский")])
                item = registry.import_csv(dict(name="Регистр", csv=content,
                                                calendar_start="2026-09-23", calendar_end="2026-12-31"))
                self.assertEqual(item["category_counts"], {"Ведущий": 2})
                self.assertEqual(registry.get(item["id"]).recommend(ORDER)["eligible_count"], 2)
                reloaded = DatasetRegistry(directory=Path(directory), settings=Settings())
                try:
                    self.assertEqual(reloaded.get(item["id"]).recommend(ORDER)["eligible_count"], 2)
                finally:
                    reloaded.close()
            finally:
                registry.close()

    def test_large_duration_is_validation_error_not_overflow(self):
        for duration in (10 ** 400, -(10 ** 400), float("inf"), float("nan"), True):
            with self.subTest(duration_type=type(duration).__name__), self.assertRaisesRegex(ValueError, "duration_hours"):
                validate_order(ORDER | {"duration_hours": duration})

    def test_large_duration_http_400(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = DatasetRegistry(directory=Path(directory), settings=Settings())
            server = make_server(port=0, registry=registry, chat=ChatService(settings=Settings()))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for path, body in (("/recommendations", ORDER | {"duration_hours": 10 ** 400}),
                                   ("/chat", {"message": "Алматы", "slots": ORDER | {"duration_hours": 10 ** 400}})):
                    request = Request("http://127.0.0.1:{}{}".format(server.server_port, path),
                                      data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(request, timeout=5)
                    self.assertEqual(caught.exception.code, 400)
                    self.assertIn("duration_hours", json.load(caught.exception)["error"])
                    caught.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
