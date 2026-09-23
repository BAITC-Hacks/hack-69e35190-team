"""Версионированные CSV-каталоги; публикация только после полной проверки."""

import hashlib
import json
import os
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from ai_layer import AIExplainer
from ai_layer.config import ROOT, Settings
from .catalog import load_catalog
from .service import RecommendationService


class DatasetRegistry:
    def __init__(self, service=None, directory=ROOT / ".local/datasets", settings=None):
        self.directory = Path(directory)
        self.settings = settings or Settings.load()
        self.lock = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="catalog-index")
        self.entries = {}
        service = service or RecommendationService(explainer=AIExplainer(settings=self.settings))
        self._register("default", "Исходный каталог", service, "provided")
        if self.directory.exists():
            for manifest in sorted(self.directory.glob("*/manifest.json")):
                info = json.loads(manifest.read_text(encoding="utf-8"))
                self._load(manifest.parent, info)

    def _register(self, dataset_id, name, service, source):
        self.entries[dataset_id] = {
            "service": service, "id": dataset_id, "name": name, "source": source,
            "index_status": "ready" if service.explainer.selector.vectors is not None else "lexical",
            "index_message": "",
        }

    def _load(self, directory, info):
        start, end = date.fromisoformat(info["calendar_start"]), date.fromisoformat(info["calendar_end"])
        path = directory / "catalog.csv"
        load_catalog(path, start, end)
        explainer = AIExplainer(path, directory / "embeddings.json", settings=self.settings)
        service = RecommendationService(path, explainer, start, end)
        self._register(directory.name, info["name"], service, "uploaded")

    def get(self, dataset_id="default"):
        if not isinstance(dataset_id, str):
            raise ValueError("dataset_id должен быть строкой")
        with self.lock:
            if dataset_id not in self.entries:
                raise ValueError("Неизвестный dataset_id; выберите каталог заново")
            return self.entries[dataset_id]["service"]

    def describe(self, dataset_id):
        with self.lock:
            service = self.get(dataset_id)
            entry = self.entries[dataset_id]
            return {key: entry[key] for key in ("id", "name", "source", "index_status", "index_message")} | {
                "profile_count": len(service.profiles),
                "synthetic_count": sum(p.synthetic for p in service.profiles),
                "csv_sha256": service.explainer.catalog.digest,
                "calendar_start": str(service.calendar_start), "calendar_end": str(service.calendar_end),
                "cities": list(service.cities), "categories": sorted(service.categories.values()),
                "category_counts": dict(Counter(category for profile in service.profiles
                                                 for category in set(profile.categories))),
                "event_types": list(service.event_types), "languages": list(service.languages),
            }

    def list(self):
        with self.lock:
            return [self.describe(key) for key in self.entries]

    def import_csv(self, data):
        if not isinstance(data, dict):
            raise ValueError("Нужен JSON-объект")
        name, content = data.get("name"), data.get("csv")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise ValueError("Название каталога: от 1 до 100 символов")
        if not isinstance(content, str) or not 1 <= len(content.encode("utf-8")) <= 2_000_000:
            raise ValueError("CSV должен быть UTF-8 текстом размером до 2 МБ")
        try:
            start = date.fromisoformat(data["calendar_start"])
            end = date.fromisoformat(data["calendar_end"])
            if start > end or (end - start).days > 3660:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ValueError("Укажите calendar_start и calendar_end (YYYY-MM-DD), период до 10 лет") from None
        digest = hashlib.sha256((str(start) + str(end) + content).encode("utf-8")).hexdigest()
        dataset_id = "csv-" + digest[:24]
        with self.lock:
            if dataset_id in self.entries:
                return self.describe(dataset_id)
            if len(self.entries) >= 50:
                raise ValueError("Лимит локального демо — 50 каталогов")
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".import-", dir=self.directory) as tmp:
                staged = Path(tmp) / dataset_id
                staged.mkdir()
                path = staged / "catalog.csv"
                path.write_text(content, encoding="utf-8")
                load_catalog(path, start, end)
                info = {"name": name.strip(), "calendar_start": str(start), "calendar_end": str(end)}
                (staged / "manifest.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
                target = self.directory / dataset_id
                os.replace(staged, target)
                self._load(target, info)
        self.index(dataset_id)
        return self.describe(dataset_id)

    def index(self, dataset_id):
        with self.lock:
            service = self.get(dataset_id)
            entry = self.entries[dataset_id]
            if entry["index_status"] in ("indexing", "ready"):
                return self.describe(dataset_id)
            if service.explainer.selector.embedder is None:
                entry["index_message"] = "Нет ключа API: работает локальный поиск фрагментов"
                return self.describe(dataset_id)
            entry["index_status"] = "indexing"
            entry["index_message"] = "Индекс строится в фоне; подбор уже доступен"
            self.worker.submit(self._build, dataset_id)
            return self.describe(dataset_id)

    def _build(self, dataset_id):
        try:
            self.get(dataset_id).explainer.prepare_index()
            status, message = "ready", "Семантический индекс готов"
        except Exception:
            # Не возвращаем исключение провайдера: оно может содержать приватный ответ.
            status, message = "degraded", "Индекс недоступен; работает локальный поиск. Можно повторить индексацию"
        with self.lock:
            self.entries[dataset_id].update(index_status=status, index_message=message)

    def close(self):
        self.worker.shutdown(wait=True)
