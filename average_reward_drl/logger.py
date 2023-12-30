from statistics import mean
from typing import Dict, List


class Logger(object):
    def __init__(self) -> None:
        self._logs: Dict[str, List[float]] = {}

    def log(self, key: str, value: float) -> None:
        if key not in self._logs:
            self._logs[key] = []
        self._logs[key].append(value)

    def flush(self) -> dict:
        data = {k: mean(v) for k, v in self._logs.items()}
        self._logs = {}
        return data

    def __str__(self) -> str:
        return str(self._logs)
