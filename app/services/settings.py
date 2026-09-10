from __future__ import annotations

from threading import RLock

from ..schemas import ChatSettings


class SettingsStore:
    def __init__(self) -> None:
        self._settings = ChatSettings()
        self._lock = RLock()

    def get(self) -> ChatSettings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def replace(self, settings: ChatSettings) -> ChatSettings:
        with self._lock:
            self._settings = settings.model_copy(deep=True)
            return self.get()

