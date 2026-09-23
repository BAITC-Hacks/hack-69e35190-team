"""Локальный HTTP-сервер: фронт и JSON API на одном адресе."""

import argparse
import csv
import json
import logging
import socket
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ai_layer.config import DEFAULT_CSV, ROOT

from .service import RecommendationService
from .chat import ChatService
from .datasets import DatasetRegistry

FRONTEND_FILE = ROOT / "frontend/index.html"
MAX_REQUEST_BYTES = 16384


def make_server(host="127.0.0.1", port=8000, service=None, registry=None, chat=None):
    registry = registry or DatasetRegistry(service=service)
    service = registry.get()
    chat = chat or ChatService()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                body = FRONTEND_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/app.js":
                body = (ROOT / "frontend/app.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/health":
                self.send_json(200, {"status": "ok", "catalog_size": len(service.profiles)})
            elif path == "/datasets":
                self.send_json(200, {"datasets": registry.list()})
            else:
                self.send_json(404, {"error": "Маршрут не найден"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/recommendations", "/chat", "/datasets", "/datasets/index"):
                self.send_json(404, {"error": "Маршрут не найден"})
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_json(403, {"error": "Откройте интерфейс на адресе этого сервера"})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Нужен Content-Type: application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                maximum = 2_100_000 if path == "/datasets" else MAX_REQUEST_BYTES
                if not 0 < length <= maximum:
                    raise ValueError("Превышен допустимый размер JSON")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("Нужен JSON-объект")
                if path == "/datasets":
                    response = registry.import_csv(body)
                elif path == "/datasets/index":
                    response = registry.index(body.get("dataset_id", "default"))
                else:
                    dataset_id = body.pop("dataset_id", "default")
                    selected = registry.get(dataset_id)
                    response = chat.respond(body, selected) if path == "/chat" else selected.recommend(body)
                    response["dataset_id"] = dataset_id
            except (ValueError, UnicodeError, csv.Error) as error:
                self.send_json(400, {"error": str(error)})
                return
            except socket.timeout:
                self.send_json(408, {"error": "Время чтения запроса истекло"})
                return
            except Exception as error:
                incident = uuid.uuid4().hex[:12]
                logging.error("request=%s exception=%s", incident, type(error).__name__)
                self.send_json(500, {"error": "Внутренняя ошибка сервера; повторите запрос", "request_id": incident})
                return
            self.send_json(200, response)

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def server_close(self):
            super().server_close()
            registry.close()

    return Server((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Подбор подрядчиков — фронт и API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    service = RecommendationService(catalog_csv=args.catalog)
    registry = DatasetRegistry(service=service)
    for item in registry.list():
        registry.index(item["id"])
    server = make_server(args.host, args.port, registry=registry)
    print("Откройте http://{}:{}".format(args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
