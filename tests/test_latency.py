import io
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request

from ai_layer import AIExplainer
from ai_layer.config import Settings
from ai_layer.embeddings import OpenAIEmbeddings
from ai_layer.http_json import load_json
from ai_layer.intent import IntentExtractor
from backend.chat import ChatService
from backend.service import RecommendationService


class SlowJSONHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = b'{"data":[{"index":0,"embedding":[1.0,0.0]}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # Каждый байт приходит раньше socket timeout, но всё тело занимает >1 с.
        try:
            for byte in body:
                if self.server.stopped.wait(0.03):
                    break
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *args):
        pass


class LatencyTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SlowJSONHandler)
        self.server.stopped = threading.Event()
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.endpoint = "http://127.0.0.1:{}".format(self.server.server_port)
        self.settings = Settings()
        self.service = RecommendationService(explainer=AIExplainer(settings=self.settings))

    def tearDown(self):
        self.server.stopped.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def test_streaming_parser_timeout_returns_local_question(self):
        extractor = IntentExtractor(self.settings, timeout=0.15)
        extractor.endpoint = self.endpoint
        started = time.monotonic()
        response = ChatService(settings=self.settings, extractor=extractor).respond(
            {"message": "Алматы"}, self.service)
        self.assertLess(time.monotonic() - started, 0.8)
        self.assertEqual(response["slots"], {"city": "Алматы"})
        self.assertEqual(response["missing_field"], "category")
        self.assertTrue(response["warnings"])

    def test_streaming_embeddings_fall_back_once_for_three_cards(self):
        embedder = OpenAIEmbeddings("test-key", dimensions=2, timeout=0.15)
        embedder.endpoint = self.endpoint
        selector = self.service.explainer.selector
        selector.embedder = embedder
        selector.vectors = [[1.0, 0.0] for _ in selector.fragments]
        order = dict(city="Алматы", event_date="2026-10-10", event_type="корпоратив",
                     category="Ведущий", budget_kzt=1500000)
        with patch.object(embedder, "embed", wraps=embedder.embed) as call:
            started = time.monotonic()
            response = self.service.recommend(order)
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(len(response["cards"]), 3)
        self.assertTrue(all(card["explanation"] for card in response["cards"]))

    def test_unresponsive_connection_is_bounded_without_waiting_for_worker(self):
        blocked = threading.Event()

        def unresponsive(*args, **kwargs):
            blocked.wait(2)
            return io.BytesIO(b'{}')

        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                load_json(Request(self.endpoint), timeout=0.1, opener=unresponsive)
            self.assertLess(time.monotonic() - started, 0.6)
        finally:
            blocked.set()

    def test_successful_embedding_body_keeps_vector_order(self):
        body = {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                         {"index": 0, "embedding": [1.0, 0.0]}]}
        with patch("ai_layer.embeddings.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
            vectors = OpenAIEmbeddings("test-key", dimensions=2).embed(["one", "two"])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
