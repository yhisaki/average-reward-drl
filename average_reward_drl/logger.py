from statistics import mean
from typing import Dict, List, Union

import torch


class Logger(object):
    def __init__(self) -> None:
        self._logs: Dict[str, List[float]] = {}

    def log(self, key: str, value: Union[float, torch.Tensor]) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if key not in self._logs:
            self._logs[key] = []
        self._logs[key].append(value)

    def flush(self) -> dict:
        data = {k: mean(v) for k, v in self._logs.items()}
        self._logs = {}
        return data

    def __str__(self) -> str:
        return str(self._logs)
