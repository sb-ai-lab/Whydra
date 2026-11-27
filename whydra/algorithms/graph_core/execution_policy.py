"""Online, bounded cost estimates for process work at immutable barriers.

Learning uses completed application work, never extra CI evaluations. Work
units describe a phase (including its conditioning order), not variable count.
The caller still owns chunk construction and graph commits.
"""

import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite, log2


@dataclass
class ProcessPlan:
    """Worker allocation and cost estimates for one process-pool operation."""

    jobs: int
    requested_jobs: int
    family: str
    work: float
    warm: bool
    record: dict
    model: "ProcessCostModel"


class ProcessCostModel:
    """EWMA latency rates, bounded by phase/work bucket/executor count.

    Defaults are conservative priors, not claims about the host. A phase learns
    its own rates as soon as it completes. Explicit ``processes`` requests keep
    their configured parallelism; only ``auto`` chooses a smaller executor.
    """

    def __init__(self, max_entries=256, cold_seconds=None):
        self.max_entries = max_entries
        self.cold_seconds = (
            (2.0 if sys.platform == "win32" else 0.15)
            if cold_seconds is None
            else cold_seconds
        )
        self._samples = OrderedDict()
        self._serial_rates = OrderedDict()
        self._startup = {}
        self._lock = threading.RLock()

    @staticmethod
    def _bucket(work):
        return int(log2(max(1.0, work)))

    def choose(
        self,
        work,
        requested_jobs,
        *,
        family,
        backend="auto",
        tasks=None,
        min_work: float = 0,
        pool_states=None,
        stats=None,
    ):
        """Choose a worker count using estimated work and measured execution costs."""
        work = float(work)
        if not isfinite(work) or work < 0:
            msg = "Process work estimate must be finite and nonnegative"
            raise ValueError(msg)
        if backend not in ("auto", "processes", "threads"):
            msg = "Unknown process policy backend"
            raise ValueError(msg)
        limit = max(1, int(requested_jobs))
        if tasks is not None:
            limit = min(limit, max(1, int(tasks)))
        states = pool_states or {}
        predictions = {}
        with self._lock:
            # The prior includes Python enumeration and bookkeeping around a CI.
            rate = self._serial_rates.get(family, (8e-6, 0))[0]
            serial = max(1e-6, work * rate)
            sample = self._samples.get((family, self._bucket(work), 1))
            if sample is not None:
                serial = max(1e-6, sample[0] * work)
            predictions[1] = serial
            if limit == 1 or work < min_work or work == 0 or backend == "threads":
                jobs, reason = 1, "single_worker" if limit == 1 else "insufficient_work"
            elif backend == "processes":
                jobs, reason = limit, "explicit_processes"
            else:
                candidates = {limit}
                value = 2
                while value <= limit:
                    candidates.add(value)
                    value *= 2
                candidates.update(
                    j
                    for j, state in states.items()
                    if 1 < j <= limit and state.get("warm")
                )
                for candidate in sorted(candidates):
                    state = states.get(candidate, {})
                    warm = state.get("warm", False)
                    busy = state.get("users", 0)
                    observed = self._samples.get(
                        (family, self._bucket(work), candidate)
                    )
                    estimate = (
                        observed[0] * work
                        if observed is not None
                        else serial / (candidate * 0.8) + 0.0015 + candidate * 0.0004
                    )
                    if not warm:
                        estimate += self._startup.get(candidate, self.cold_seconds)
                    # Do not assume that a pool already leased by another fit is idle.
                    estimate *= 1 + busy
                    predictions[candidate] = estimate
                best = min(predictions, key=predictions.__getitem__)
                jobs = best if predictions[best] < serial * 0.9 else 1
                reason = (
                    "predicted_process_gain" if jobs > 1 else "predicted_serial_faster"
                )
        warm = bool(states.get(jobs, {}).get("warm", False)) if jobs > 1 else True
        record = {
            "family": family,
            "estimated_work": work,
            "requested_jobs": requested_jobs,
            "selected_jobs": jobs,
            "requested_backend": backend,
            "pool_warm": warm,
            "reason": reason,
            "predicted_seconds": {str(k): v for k, v in predictions.items()},
        }
        if stats is not None:
            records = stats.setdefault("process_policy", [])
            records.append(record)
            if len(records) > 64:
                del records[:-64]
        return ProcessPlan(jobs, requested_jobs, family, work, warm, record, self)

    def observe(self, plan, seconds):
        """Update the cost estimate with the elapsed time for a completed plan."""
        if (
            plan.model is not self
            or plan.work <= 0
            or not isfinite(seconds)
            or seconds <= 0
        ):
            return
        plan.record["observed_seconds"] = seconds
        with self._lock:
            sample_seconds = seconds
            if plan.jobs > 1 and not plan.warm:
                # A cold sample must not contaminate steady-state throughput.
                warm_prediction = (
                    plan.record["predicted_seconds"].get("1", 0) / (plan.jobs * 0.8)
                    + 0.0015
                    + plan.jobs * 0.0004
                )
                startup = max(0.0, seconds - warm_prediction)
                old = self._startup.get(plan.jobs)
                self._startup[plan.jobs] = (
                    startup if old is None else 0.75 * old + 0.25 * startup
                )
                plan.record["estimated_startup_seconds"] = startup
                return
            key = (plan.family, self._bucket(plan.work), plan.jobs)
            value, count = self._samples.pop(key, (sample_seconds / plan.work, 0))
            self._samples[key] = (
                0.75 * value + 0.25 * sample_seconds / plan.work,
                count + 1,
            )
            while len(self._samples) > self.max_entries:
                self._samples.popitem(last=False)
            if plan.jobs == 1:
                value, count = self._serial_rates.pop(
                    plan.family, (seconds / plan.work, 0)
                )
                self._serial_rates[plan.family] = (
                    0.75 * value + 0.25 * seconds / plan.work,
                    count + 1,
                )
                while len(self._serial_rates) > self.max_entries:
                    self._serial_rates.popitem(last=False)


PROCESS_COST_MODEL = ProcessCostModel()
