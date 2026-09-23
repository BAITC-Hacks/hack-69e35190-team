"""Пользовательские сокращения и опечатки работают без внешнего AI."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

from ai_layer import AIExplainer
from ai_layer.config import Settings
from backend.catalog import Contractor
from backend.chat import ChatService
from backend.datasets import DatasetRegistry
from backend.service import RecommendationService


BASE = {
    "city": "Алматы", "category": "Ведущий", "event_type": "свадьба",
    "event_date": "2026-10-10", "budget_kzt": 3000000,
}


class ChatLanguageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RecommendationService(explainer=AIExplainer(settings=Settings()))

    def setUp(self):
        self.chat = ChatService(settings=Settings())

    def respond(self, message, slots=None, service=None):
        return self.chat.respond(
            {"message": message, "slots": slots or {}}, service or self.service)

    def test_short_city_and_category_answers(self):
        city = self.respond("ала")
        self.assertEqual(city["slots"], {"city": "Алматы"})
        self.assertEqual(city["missing_field"], "category")
        category = self.respond("ВЕД!", city["slots"])
        self.assertEqual(category["slots"], {"city": "Алматы", "category": "Ведущий"})
        self.assertEqual(category["missing_field"], "event_type")
        self.assertEqual(category["parser"], "local")

    def test_user_sentence_keeps_fields_until_category_answer(self):
        response = self.respond("так надо в алмате свадба бюджет 3млн на 22 октябр")
        expected = {
            "city": "Алматы", "event_type": "свадьба",
            "budget_kzt": 3000000, "event_date": "2026-10-22",
        }
        self.assertEqual(response["slots"], expected)
        self.assertEqual(response["missing_field"], "category")
        self.assertIsNone(response["result"])

        completed = self.respond("вед", response["slots"])
        self.assertEqual(completed["slots"], expected | {"category": "Ведущий"})
        self.assertIsNone(completed["missing_field"])
        self.assertIsNotNone(completed["result"])
        self.assertEqual(completed["parser"], "local")

    def test_abbreviations_in_one_message(self):
        response = self.respond("алм вед корп 22 окт бюджет 1,5млн")
        self.assertEqual(response["slots"], {
            "city": "Алматы", "category": "Ведущий", "event_type": "корпоратив",
            "event_date": "2026-10-22", "budget_kzt": 1500000,
        })
        self.assertIsNone(response["missing_field"])

    def test_common_typo_and_inflected_words(self):
        response = self.respond("нужен ведуший в Алмате на свадбу 22 окт бюджет 3 млн")
        self.assertEqual(response["slots"], {
            "city": "Алматы", "category": "Ведущий", "event_type": "свадьба",
            "event_date": "2026-10-22", "budget_kzt": 3000000,
        })
        self.assertIsNotNone(response["result"])

    def test_date_time_duration_and_budget_do_not_overlap(self):
        response = self.respond("ала вед свадба 22 окт 2026 в 15:00 на 3ч бюджет 3млн")
        self.assertEqual(response["slots"], {
            "city": "Алматы", "category": "Ведущий", "event_type": "свадьба",
            "event_date": "2026-10-22", "duration_hours": 3, "budget_kzt": 3000000,
        })
        self.assertTrue(any("время" in warning for warning in response["warnings"]))

    def test_decimal_money_with_unit_is_not_a_numeric_date(self):
        response = self.respond("ала вед свадба 22 окт бюджет 1.5 млн")
        self.assertEqual(response["slots"], BASE | {
            "event_date": "2026-10-22", "budget_kzt": 1500000,
        })

    def test_date_after_do_does_not_replace_budget(self):
        response = self.respond("до 22 октября", BASE)
        self.assertEqual(response["slots"], BASE | {"event_date": "2026-10-22"})

    def test_december_abbreviation_does_not_change_category(self):
        response = self.respond("на 22 дек", BASE)
        self.assertEqual(response["slots"], BASE | {"event_date": "2026-12-22"})

    def test_shorthand_does_not_match_inside_unrelated_words(self):
        for message in ("ведро", "алабай", "подвал", "22 ноябрябрь", "фотоаппарат",
                        "корпус", "дребезг", "флорбол", "декан"):
            with self.subTest(message=message):
                response = self.respond(message)
                self.assertEqual(response["slots"], {})
                self.assertIsNone(response["result"])

    def test_exact_imported_vocabulary_and_ambiguous_prefix(self):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(Contractor.__dataclass_fields__))
        writer.writeheader()
        for number, category in enumerate(("Световое шоу", "Световой художник", "Скрипач", "Скрипка"), 1):
            writer.writerow({
                "id": "LIGHT-{}".format(number), "anon_name": "Мастер {}".format(number),
                "categories": category, "city": "Караганда", "price_from_kzt": 30000,
                "event_formats": "выпускной", "languages": "испанский", "max_hours": "null",
                "busy_dates": "2027-05-02", "description": "Создаёт световые рисунки на выпускном.",
                "synthetic": "true", "city_imputed": "false", "price_imputed": "false",
            })
        with tempfile.TemporaryDirectory() as directory:
            registry = DatasetRegistry(self.service, Path(directory), settings=Settings())
            try:
                imported = registry.import_csv({
                    "name": "Новые услуги", "csv": output.getvalue(),
                    "calendar_start": "2027-05-01", "calendar_end": "2027-05-31",
                })
                service = registry.get(imported["id"])
                exact = self.respond(
                    "Световой художник Караганда выпускной 1 мая бюджет 50000", service=service)
                self.assertEqual(exact["slots"], {
                    "city": "Караганда", "category": "Световой художник", "event_type": "выпускной",
                    "event_date": "2027-05-01", "budget_kzt": 50000,
                })
                self.assertEqual(exact["result"]["cards"][0]["id"], "LIGHT-2")
                ambiguous = self.respond("свет", service=service)
                self.assertNotIn("category", ambiguous["slots"])
                # «скрипак» на одну опечатку отличается от обеих новых категорий.
                ambiguous_typo = self.respond("скрипак", service=service)
                self.assertNotIn("category", ambiguous_typo["slots"])
            finally:
                registry.close()


if __name__ == "__main__":
    unittest.main()
