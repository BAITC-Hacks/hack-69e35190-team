"""Ограничивает всё ожидание опционального HTTP-запроса, включая чтение тела."""

import json
import math
import threading
import time


# Зависшие DNS/сетевые вызовы не должны создавать неограниченное число потоков.
_WORKERS = threading.BoundedSemaphore(8)
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def load_json(request, *, timeout, opener):
    """Socket timeout защищает от пауз, deadline — от медленной передачи по байту.

    Сетевой поток daemon: даже если ОС задержит DNS, запрос пользователя перейдёт
    на локальный ответ. При занятых worker-слотах также сразу включается fallback.
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Таймаут должен быть положительным числом")
    deadline = time.monotonic() + timeout
    if not _WORKERS.acquire(blocking=False):
        raise TimeoutError("Сервис AI занят")
    completed = threading.Event()
    outcome = []

    def fetch():
        try:
            with opener(request, timeout=timeout) as response:
                # HTTPResponse.read() ждёт весь размер даже при постоянном потоке
                # байтов; read1() возвращает одну порцию и позволяет проверить срок.
                read = getattr(response, "read1", response.read)
                chunks, size = [], 0
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Истёк срок ожидания AI")
                    chunk = read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > _MAX_RESPONSE_BYTES:
                        raise ValueError("Ответ AI слишком большой")
                    chunks.append(chunk)
                outcome.append((True, json.loads(b"".join(chunks))))
        except Exception as error:
            outcome.append((False, error))
        finally:
            _WORKERS.release()
            completed.set()

    try:
        threading.Thread(target=fetch, name="ai-http", daemon=True).start()
    except Exception:
        _WORKERS.release()
        raise
    if not completed.wait(max(0, deadline - time.monotonic())):
        raise TimeoutError("Истёк срок ожидания AI")
    succeeded, value = outcome[0]
    if not succeeded:
        raise value
    return value
