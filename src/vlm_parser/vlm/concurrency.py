from __future__ import annotations

from threading import Semaphore
from types import TracebackType


class GlobalVlmLimiter:
    def __init__(self, max_concurrency: int):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._semaphore = Semaphore(max_concurrency)

    def __enter__(self) -> "GlobalVlmLimiter":
        self._semaphore.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._semaphore.release()
