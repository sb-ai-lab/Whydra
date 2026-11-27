"""Ordered public CI transactions retained alongside compact worker batches."""

from concurrent.futures import Future


class RetryCI(Exception):
    """A speculative owner released an unconsumed request; retry normally."""


def ordered_batch(test, requests):
    """Evaluate CI requests in order, batching uncached built-in Fisher-Z requests."""
    from . import independence_tests as ci

    if not ci._builtin_fisherz(test):
        return [test(x, y, conditions) for x, y, conditions in requests]
    output, pending = [], []

    def flush():
        if not pending:
            return
        remaining = {key for _x, _y, _conditions, key in pending}
        try:
            values = ci._fisher_batch_results(
                test.corr_matrix,
                test.n_samples,
                [(x, y, conditions) for x, y, conditions, _ in pending],
            )
            for (x, y, conditions, key), value in zip(pending, values, strict=False):
                with test._lock:
                    test.cache_misses += 1
                    waiter = test._inflight.pop(key, None)
                    remaining.discard(key)
                    if isinstance(value, BaseException):
                        if isinstance(waiter, Future):
                            waiter.set_exception(value)
                        raise value
                    test._store(key, value, (x, y))
                    value = test._decision_value(x, y, conditions, value)
                    if isinstance(waiter, Future):
                        waiter.set_result(value)
                output.append(value)
        finally:
            with test._lock:
                for key in remaining:
                    waiter = test._inflight.pop(key, None)
                    if isinstance(waiter, Future):
                        waiter.set_exception(RetryCI())
            pending.clear()

    iterator = iter(requests)
    while True:
        try:
            x, y, conditions = next(iterator)
            x, y, conditions, key = test._request(x, y, conditions)
        except StopIteration:
            flush()
            return output
        except BaseException:
            flush()
            raise
        with test._lock:
            ready = key in test._cache or key in test._inflight
            if not ready:
                test._inflight[key] = None
                pending.append((x, y, conditions, key))
        if ready:
            flush()
            output.append(test(x, y, conditions))
        elif len(pending) >= ci._CI_BATCH_QUERIES:
            flush()
