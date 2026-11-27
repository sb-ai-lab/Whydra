import os
import time
from collections import deque
from contextlib import contextmanager


class Profiler:
    """Collect and retain the most recent profiling records in a shared instance."""

    records: deque
    enabled: bool
    _instance = None

    def __new__(cls):
        """Return the shared profiler, creating its record buffer on first use."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.records = deque(maxlen=10000)
            cls._instance.enabled = os.environ.get("WHYDRA_PROFILE", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        return cls._instance

    @contextmanager
    def time_block(self, name, tags=None):
        """Time a block of code with a context manager."""
        if not self.enabled:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.records.append(
                {"name": name, "duration": duration, "tags": tags or {}}
            )

    def get_stats(self):
        """Return the current profiling records as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame.from_records(list(self.records))

    def clear(self):
        """Clear the stored profiling records."""
        self.records.clear()


# Global instance for convenience.
profiler = Profiler()
