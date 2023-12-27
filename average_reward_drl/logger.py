class Logger(object):
    def __init__(self) -> None:
        self._logs = {}

    def log(self, key: str, value: float, method: str) -> None:
        ...

    def flush(self) -> None:
        pass
