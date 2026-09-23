"""Регрессии аудита: явные ограничения не теряются при частичном разборе."""

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
from backend.chat import ChatService, local_updates, vocabulary
from backend.datasets import DatasetRegistry
from backend.service import RecommendationService


BASE = dict(city="Алматы", event_date="2026-10-10", event_type="корпоратив",
            category="Ведущий", budget_kzt=1500000)


class ChatRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RecommendationService(explainer=AIExplainer(settings=Settings()))

    def setUp(self):
        self.chat = ChatService(settings=Settings())

    def respond(self, message, slots=None, **kwargs):
        return self.chat.respond(dict(message=message, slots=BASE if slots is None else slots, **kwargs), self.service)

    def test_corrections_choose_the_replacement_and_preserve_other_fields(self):
        cases = [("Не Алматы, а Астана", {"city": "Астана"}),
                 ("Не 10 октября, а 11 октября", {"event_date": "2026-10-11"}),
                 ("Не фотограф, а ведущий", {"category": "Ведущий"}),
                 ("Не 500 тыс, а 800 тыс", {"budget_kzt": 800000}),
                 ("Не 1.5 млн, а 800 тыс", {"budget_kzt": 800000}),
                 ("Не 10.10.26, а 11.10.26", {"event_date": "2026-10-11"}),
                 ("Не 6 часов, а 3 часа", {"duration_hours": 3}),
                 ("Не русский, а казахский", {"language": "казахский"}),
                 ("Не Алматы а Астана и не 10 октября, а 11 октября",
                  {"city": "Астана", "event_date": "2026-10-11"})]
        for message, changes in cases:
            with self.subTest(message=message):
                response = self.respond(message)
                self.assertEqual(response["slots"], BASE | changes)
                self.assertIsNone(response["missing_field"])

    def test_decimal_partial_updates_are_not_dates(self):
        for message, update in (("бюджет 1.5 млн", {"budget_kzt": 1500000}),
                                ("на 6.5 часов", {"duration_hours": 6.5}),
                                ("на 90.5 минут", {"duration_hours": 90.5 / 60})):
            with self.subTest(message=message):
                response = self.respond(message)
                self.assertEqual(response["slots"], BASE | update)
                self.assertEqual(response["pending_fields"], [])
                self.assertIsNotNone(response["result"])

    def test_maximum_budget_is_not_lost_when_language_is_recognized(self):
        for message in ("Бюджет максимум 500000, язык русский", "Бюджет не более 500000, язык русский",
                        "Не более 500000, язык русский", "Бюджет до 500 000, язык русский"):
            with self.subTest(message=message):
                response = self.respond(message)
                self.assertEqual(response["slots"]["budget_kzt"], 500000)
                self.assertEqual(response["slots"]["language"], "русский")
                self.assertEqual([card["id"] for card in response["result"]["cards"]], ["HK-88430"])

    def test_hours_minutes_are_added_before_strict_filtering(self):
        for message in ("на 6 часов 30 минут", "на 6 ч 30 мин", "на 6ч30мин", "на 6 ч. 30 мин.", "на 6ч.30мин."):
            with self.subTest(message=message):
                response = self.respond(message)
                self.assertEqual(response["slots"]["duration_hours"], 6.5)
                self.assertTrue(response["result"]["cards"])
                self.assertTrue({"HK-88430", "HK-29829"}.isdisjoint(
                    card["id"] for card in response["result"]["cards"]))
        self.assertEqual(self.respond("на 90 минут")["slots"]["duration_hours"], 1.5)

    def test_start_time_is_not_duration_and_is_explicitly_warned(self):
        for message in ("в 15 часов на 3 часа", "в 15 часов 30 минут на 3 часа",
                        "в 15 часов и 30 минут на 3 часа", "в 15:30 на 3 часа"):
            with self.subTest(message=message):
                response = self.respond(message)
                self.assertEqual(response["slots"]["duration_hours"], 3)
                self.assertEqual(response["result"]["status"], "matched")
                self.assertTrue(any("время начала" in warning for warning in response["warnings"]))

    def test_two_digit_year_is_never_silently_replaced_by_calendar_year(self):
        for message in ("на 25.10.27", "на 25/10/27", "на 25 октября 27 года", "на 25 октября 2027"):
            with self.subTest(message=message):
                self.assertEqual(local_updates(message, BASE, vocabulary(self.service))["event_date"], "2027-10-25")
                with self.assertRaisesRegex(ValueError, "Календарь известен"):
                    self.respond(message)
        self.assertEqual(self.respond("25.10.26")["slots"]["event_date"], "2026-10-25")

    def test_both_optional_constraints_can_be_cleared_together(self):
        response = self.respond("Язык любой, длительность не важна", BASE | {
            "language": "русский", "duration_hours": 12})
        self.assertIsNone(response["slots"]["language"])
        self.assertIsNone(response["slots"]["duration_hours"])
        self.assertEqual(response["result"]["status"], "matched")

    def test_unsupported_language_blocks_recommendation_without_losing_other_updates(self):
        for message in ("Астана, бюджет 500000, на немецком", "Астана, бюджет 500000, язык немецкий"):
            with self.subTest(message=message):
                response = self.respond(message, BASE | {"language": "русский"})
                self.assertEqual(response["slots"], (BASE | {"city": "Астана", "budget_kzt": 500000}))
                self.assertIsNone(response["result"])
                self.assertEqual(response["missing_field"], "clarification")
                self.assertEqual(response["pending_fields"], ["language"])
                self.assertIn("язык", response["prompt"])
                self.assertIn("русский", response["choices"])

    def test_pending_constraint_survives_unrelated_turn_until_explicit_resolution(self):
        first = self.respond("на немецком")
        second = self.respond("бюджет 500000", first["slots"], pending_fields=first["pending_fields"])
        self.assertEqual(second["pending_fields"], ["language"])
        self.assertIsNone(second["result"])
        final = self.respond("язык не важен", second["slots"], pending_fields=second["pending_fields"])
        self.assertEqual(final["pending_fields"], [])
        self.assertEqual(final["slots"]["budget_kzt"], 500000)
        self.assertIsNone(final["slots"]["language"])
        self.assertEqual(final["result"]["status"], "matched")
        for value in ("language", ["unknown"], [1]):
            with self.assertRaisesRegex(ValueError, "pending_fields"):
                self.respond("русский", pending_fields=value)

    def test_partially_understood_change_requires_resolution(self):
        for message, pending in (("Язык русский, бюджет полмиллиона", "budget_kzt"),
                                 ("Язык русский, длительность полдня", "duration_hours"),
                                 ("Не фотограф", "category")):
            with self.subTest(message=message):
                response = self.respond(message)
                self.assertIsNone(response["result"])
                self.assertEqual(response["pending_fields"], [pending])

    def test_llm_is_called_for_unparsed_constraint_despite_other_updates(self):
        class Extractor:
            called = False

            def extract(self, *args):
                self.called = True
                return {"budget_kzt": 500000}

        extractor = Extractor()
        response = ChatService(settings=Settings(), extractor=extractor).respond(
            dict(message="Бюджет полмиллиона, язык русский", slots=BASE), self.service)
        self.assertTrue(extractor.called)
        self.assertEqual(response["slots"]["budget_kzt"], 500000)
        self.assertEqual(response["slots"]["language"], "русский")
        self.assertEqual(response["pending_fields"], [])

    def test_llm_cannot_clear_pending_language_when_only_budget_changes(self):
        class IncorrectExtractor:
            def extract(self, *args):
                return {"language": None}

        response = ChatService(settings=Settings(), extractor=IncorrectExtractor()).respond(
            dict(message="Бюджет 500000", slots=BASE, pending_fields=["language"]), self.service)
        self.assertIsNone(response["result"])
        self.assertEqual(response["pending_fields"], ["language"])
        self.assertNotIn("language", response["slots"])
        self.assertEqual(response["slots"]["budget_kzt"], 500000)

    def test_imported_language_inflection_uses_dynamic_vocabulary(self):
        vocab = vocabulary(self.service) | {"language": ["испанский"]}
        self.assertEqual(local_updates("на испанском", {}, vocab), {"language": "испанский"})

    def test_http_chat_clarifies_unknown_language_and_rejects_year_outside_calendar(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = DatasetRegistry(self.service, Path(directory), settings=Settings())
            server = make_server(port=0, registry=registry, chat=self.chat)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def post(message, **kwargs):
                request = Request("http://127.0.0.1:{}/chat".format(server.server_port),
                                  json.dumps(dict(message=message, slots=BASE, **kwargs)).encode(),
                                  headers={"Content-Type": "application/json"})
                return urlopen(request, timeout=5)
            try:
                with post("бюджет 500000, на немецком") as result:
                    response = json.load(result)
                    self.assertEqual(result.status, 200)
                    self.assertIsNone(response["result"])
                    self.assertEqual(response["pending_fields"], ["language"])
                with self.assertRaises(HTTPError) as caught:
                    post("25.10.27")
                self.assertEqual(caught.exception.code, 400)
                caught.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                registry.close()


if __name__ == "__main__":
    unittest.main()
