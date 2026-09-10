from __future__ import annotations

from threading import RLock

from ..schemas import ChatExperimentSettings


class ExperimentSettingsStore:
    """Process-local settings applied to subsequent agent calls."""

    def __init__(self) -> None:
        self._settings = ChatExperimentSettings()
        self._lock = RLock()

    def get(self) -> ChatExperimentSettings:
        with self._lock:
            return self._settings.model_copy(deep=True)

    def replace(self, settings: ChatExperimentSettings) -> ChatExperimentSettings:
        with self._lock:
            self._settings = settings.model_copy(deep=True)
            return self.get()


# Compatibility name retained for callers of the first harness version.
SettingsStore = ExperimentSettingsStore
