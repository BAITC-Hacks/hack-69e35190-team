import csv
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_layer import AIExplainer
from ai_layer.config import Settings
from ai_layer.intent import IntentExtractor
from backend.api import make_server
from backend.catalog import Contractor
from backend.chat import ChatService
from backend.datasets import DatasetRegistry
from backend.service import RecommendationService

SCREENSHOT = 'фотосессия на корпоратив, в Алмате на 25 октября, в 15:00 на 3 часа, бюджет 50тыс, язык не важен'
BASE = dict(city='Алматы', event_date='2026-10-10', event_type='корпоратив', category='Ведущий', budget_kzt=1500000)


def sample_csv(**changes):
    row = dict(id='NEW-1', anon_name='Новый фотограф', categories='Фотограф', city='Караганда',
               price_from_kzt='30000', event_formats='выпускной', languages='испанский', max_hours='null',
               busy_dates='2027-05-02', description='Снимает выпускные вечера с моментальной печатью фотографий.',
               synthetic='true', city_imputed='false', price_imputed='false')
    row.update(changes)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(Contractor.__dataclass_fields__))
    writer.writeheader()
    writer.writerow(row)
    return out.getvalue()


def import_payload(**changes):
    return dict(name='Новый кейс', csv=sample_csv(**changes), calendar_start='2027-05-01', calendar_end='2027-05-31')


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.service = RecommendationService(explainer=AIExplainer(settings=Settings()))
        self.registry = DatasetRegistry(self.service, Path(self.directory.name), settings=Settings())
        self.chat = ChatService(settings=Settings())

    def tearDown(self):
        self.registry.close()
        self.directory.cleanup()

    def test_screenshot_without_api(self):
        started = time.monotonic()
        result = self.chat.respond(dict(message=SCREENSHOT), self.service)
        self.assertEqual(result['slots'], dict(city='Алматы', category='Фотограф', event_type='корпоратив',
                         event_date='2026-10-25', budget_kzt=50000, duration_hours=3, language=None))
        self.assertEqual(result['result']['status'], 'no_matches')
        self.assertEqual(result['result']['exclusion_counts']['budget'], 8)
        self.assertTrue(result['warnings'])
        self.assertLess(time.monotonic() - started, 10)

    def test_dialogue_and_changed_date(self):
        slots = {}
        for message in ['Алматы', 'Ведущий', 'корпоратив', '10 октября', '1500000']:
            response = self.chat.respond(dict(message=message, slots=slots), self.service)
            slots = response['slots']
        self.assertEqual(slots, BASE)
        first = [c['id'] for c in response['result']['cards']]
        second = self.chat.respond(dict(message='давай на 11 октября', slots=slots), self.service)
        self.assertNotEqual(first, [c['id'] for c in second['result']['cards']])
        self.assertIn('заняты на дату', second['result']['message'])

    def test_optional_fields_and_money(self):
        response = self.chat.respond(dict(message='бюджет 1,5 млн на 3,5 часа, язык не важен',
                                         slots=BASE | {'language': 'русский'}), self.service)
        self.assertEqual(response['slots']['budget_kzt'], 1500000)
        self.assertEqual(response['slots']['duration_hours'], 3.5)
        self.assertIsNone(response['slots']['language'])
        for text, expected in [('бюджет 900 000', 900000), ('50к', 50000), ('бюджет 1', 1)]:
            self.assertEqual(self.chat.respond(dict(message=text, slots=BASE), self.service)['slots']['budget_kzt'], expected)

    def test_llm_failure_falls_back(self):
        class Offline:
            def extract(self, *args):
                raise TimeoutError('provider secret')
        response = ChatService(settings=Settings(), extractor=Offline()).respond(dict(message='Алматы'), self.service)
        self.assertEqual(response['slots'], {'city': 'Алматы'})
        self.assertEqual(response['missing_field'], 'category')
        self.assertNotIn('provider secret', json.dumps(response))

    def test_llm_output_is_validated(self):
        class Malformed:
            def extract(self, *args):
                return {'duration_hours': -10, 'event_date': '2050-01-01'}
        response = ChatService(settings=Settings(), extractor=Malformed()).respond(dict(message='ещё вариант', slots=BASE), self.service)
        self.assertEqual(response['slots'], BASE)
        self.assertTrue(response['warnings'])
        self.assertIsNone(response['result'])
        self.assertEqual(response['missing_field'], 'clarification')
        for invalid in [{'duration_hours': float('nan')}, {'budget_kzt': True}, {'event_date': '2026-02-30'}]:
            with self.assertRaises(ValueError):
                self.chat.respond(dict(message='Алматы', slots=BASE | invalid), self.service)

    def test_import_isolated_persistent_and_idempotent(self):
        item = self.registry.import_csv(import_payload())
        self.assertEqual(item['profile_count'], 1)
        self.assertEqual(item['synthetic_count'], 1)
        self.assertEqual(item['index_status'], 'lexical')
        self.assertIn('Караганда', item['cities'])
        service = self.registry.get(item['id'])
        query = dict(city='Караганда', event_date='2027-05-01', event_type='выпускной',
                     category='Фотограф', budget_kzt=50000, language='испанский', duration_hours=24)
        first = service.recommend(query)
        self.assertEqual(first['cards'][0]['id'], 'NEW-1')
        self.assertIn('моментальной печатью', first['cards'][0]['explanation'])
        self.assertTrue(first['cards'][0]['synthetic'])
        self.assertEqual(service.recommend(query | {'event_date': '2027-05-02'})['status'], 'no_matches')
        self.assertEqual(len(self.registry.get().profiles), 66)
        self.assertEqual(self.registry.import_csv(import_payload())['id'], item['id'])
        self.assertEqual(len(self.registry.list()), 2)
        restarted = DatasetRegistry(self.service, Path(self.directory.name), settings=Settings())
        try:
            self.assertEqual(restarted.get(item['id']).recommend(query), first)
        finally:
            restarted.close()
        # Новая версия тех же id не перезаписывает прежний каталог.
        other = self.registry.import_csv(import_payload(price_from_kzt='90000'))
        self.assertNotEqual(other['id'], item['id'])
        self.assertEqual(self.registry.get(other['id']).recommend(query)['status'], 'no_matches')
        self.assertEqual(service.recommend(query), first)

    def test_invalid_import_never_publishes(self):
        for payload in [import_payload(price_from_kzt='nan'), import_payload(busy_dates='2028-01-01'),
                        import_payload(max_hours='nan'), import_payload(synthetic='yes'),
                        dict(name='bad', csv='wrong,header\n1,2', calendar_start='2027-01-01', calendar_end='2027-12-31')]:
            with self.assertRaises(ValueError):
                self.registry.import_csv(payload)
        self.assertEqual(len(self.registry.list()), 1)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_http_integration_and_safe_500(self):
        server = make_server(port=0, registry=self.registry, chat=self.chat)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:{}'.format(server.server_port)
        def post(path, body):
            request = Request(base + path, json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=5) as response:
                return json.load(response)
        try:
            self.assertEqual(post('/chat', {'message': SCREENSHOT})['result']['status'], 'no_matches')
            item = post('/datasets', import_payload())
            response = post('/chat', {'message': 'Фотограф Караганда выпускной 1 мая бюджет 50000', 'dataset_id': item['id']})
            self.assertEqual(response['slots']['event_date'], '2027-05-01')
            self.assertEqual(response['result']['cards'][0]['id'], 'NEW-1')
            with patch.object(self.service, 'recommend', side_effect=RuntimeError('TOP_SECRET')):
                with self.assertRaises(HTTPError) as caught:
                    post('/recommendations', BASE)
                self.assertEqual(caught.exception.code, 500)
                body = json.load(caught.exception)
                caught.exception.close()
                self.assertIn('request_id', body)
                self.assertNotIn('TOP_SECRET', json.dumps(body))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_intent_requires_verbatim_evidence(self):
        updates = {'updates': [{'field': 'city', 'value': 'Алматы', 'evidence': 'Алмате'},
                               {'field': 'budget_kzt', 'value': '999999', 'evidence': 'выдуманный бюджет'}]}
        response = io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(updates)}}]}).encode())
        with patch('ai_layer.intent.urlopen', return_value=response):
            actual = IntentExtractor(Settings()).extract('в Алмате', {}, {})
        self.assertEqual(actual, {'city': 'Алматы'})


if __name__ == '__main__':
    unittest.main()
