import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_layer import AIExplainer
from ai_layer.config import ROOT, Settings
from backend.api import make_server
from backend.service import RecommendationService, validate_order


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RecommendationService(explainer=AIExplainer(settings=Settings()))
        cls.cases = json.loads((ROOT / "examples/demo-cases.json").read_text(encoding="utf-8"))

    def test_all_demo_cases_match_csv(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                actual = self.service.recommend(case["request"])
                expected = case["expected"]
                self.assertEqual(actual["status"], expected["status"])
                self.assertEqual(actual["candidate_count"], expected["candidate_count"])
                self.assertEqual(actual["eligible_count"], expected["eligible_count"])
                self.assertEqual(actual["exclusion_counts"], expected["exclusion_counts"])
                self.assertEqual([card["id"] for card in actual["cards"]], expected["card_ids"])
                self.assertTrue(actual["message"])
                for card in actual["cards"]:
                    self.assertTrue(card["explanation"])
                    self.assertLessEqual(card["price_from_kzt"], case["request"]["budget_kzt"])

    def test_repeatable_order_and_date_switch(self):
        first = self.cases[0]["request"]
        second = self.cases[1]["request"]
        self.assertEqual(self.service.recommend(first), self.service.recommend(first))
        ids_first = [card["id"] for card in self.service.recommend(first)["cards"]]
        ids_second = [card["id"] for card in self.service.recommend(second)["cards"]]
        self.assertNotEqual(ids_first, ids_second)
        self.assertIn("заняты на дату", self.service.recommend(second)["message"])

    def test_language_and_duration_are_hard_filters(self):
        base = self.cases[0]["request"]
        result = self.service.recommend(dict(base, language="казахский", duration_hours=8))
        self.assertGreater(result["exclusion_counts"]["language"], 0)
        self.assertGreater(result["exclusion_counts"]["duration"], 0)
        for card in result["cards"]:
            profile = next(profile for profile in self.service.profiles if profile.id == card["id"])
            self.assertIn("казахский", profile.languages)
            self.assertTrue(profile.max_hours is None or profile.max_hours >= 8)

    def test_florist_null_hours_is_not_rejected(self):
        result = self.service.recommend(dict(self.cases[2]["request"], duration_hours=12))
        self.assertEqual([card["id"] for card in result["cards"]], ["HK-39372"])
        self.assertIn("не привязана к часам", result["cards"][0]["explanation"])

    def test_invalid_requests(self):
        base = self.cases[0]["request"]
        invalid = (
            {"event_date": "2027-01-01"}, {"event_date": "2026-02-30"},
            {"budget_kzt": True}, {"budget_kzt": 0.5}, {"duration_hours": -1},
            {"language": "испанский"}, {"city": "Караганда"}, {"extra": 1},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_order(dict(base, **values))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RecommendationService(explainer=AIExplainer(settings=Settings()))
        cls.server = make_server(port=0, service=cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:{}".format(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def _post(self, payload):
        request = Request(self.url + "/recommendations", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)

    def test_frontend_and_health_are_served(self):
        with urlopen(self.url + "/", timeout=5) as response:
            page = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
        self.assertIn('src="/app.js"', page)
        with urlopen(self.url + "/app.js", timeout=5) as response:
            self.assertIn("api('/chat'", response.read().decode("utf-8"))
        with urlopen(self.url + "/health", timeout=5) as response:
            self.assertEqual(json.load(response)["catalog_size"], 66)

    def test_post_end_to_end_and_empty_results(self):
        cases = json.loads((ROOT / "examples/demo-cases.json").read_text(encoding="utf-8"))
        for case in cases:
            with self.subTest(case=case["id"]):
                status, result = self._post(case["request"])
                self.assertEqual(status, 200)
                self.assertEqual(result["status"], case["expected"]["status"])
                self.assertEqual([card["id"] for card in result["cards"]], case["expected"]["card_ids"])

    def test_bad_json_is_client_error(self):
        request = Request(self.url + "/recommendations", data=b"{",
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("error", json.load(caught.exception))
        caught.exception.close()


if __name__ == "__main__":
    unittest.main()
