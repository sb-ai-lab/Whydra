import time
from contextlib import contextmanager
import pandas as pd

class Profiler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            cls._instance.records = []
        return cls._instance

    @contextmanager
    def time_block(self, name, tags=None):
        """Контекстный менеджер для замера времени."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.records.append({
                'name': name,
                'duration': duration,
                'tags': tags or {}
            })

    def get_stats(self):
        return pd.DataFrame.from_records(self.records)

    def clear(self):
        self.records = []

# Глобальный экземпляр для удобства
profiler = Profiler()