"""Read-only numerical datasets attached once per worker and dataset revision.

The coordinator owns bounded shared-memory segments. Leases cover queued tasks
as well as running ones; a failed sibling fit cannot unlink their data. Workers
cache read-only attachments by token, so tasks carry only a small descriptor.
"""

import atexit
import inspect
import os
import sys
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from multiprocessing import shared_memory

import numpy as np

_TRACK_SUPPORTED = "track" in inspect.signature(shared_memory.SharedMemory).parameters
_SHARING_SUPPORTED = _TRACK_SUPPORTED or sys.platform == "win32"
_LOCK = threading.RLock()
_ENTRIES = OrderedDict()
_BY_ID = {}
_WORKER_DATA = OrderedDict()
_MAX_ENTRIES = 8
_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SharedArray:
    """Location, shape and dtype of an array in shared memory."""

    name: str
    shape: tuple
    dtype: str
    nbytes: int


def _open_memory(**kwargs):
    if _TRACK_SUPPORTED:
        kwargs["track"] = False
    return shared_memory.SharedMemory(**kwargs)


def _drop(entry):
    token = entry["ref"].name
    _ENTRIES.pop(token, None)
    if _BY_ID.get(entry["identity"]) is entry:
        _BY_ID.pop(entry["identity"], None)
    try:
        entry["memory"].unlink()
    except FileNotFoundError:
        pass
    finally:
        entry["memory"].close()


def _trim():
    total = sum(e["ref"].nbytes for e in _ENTRIES.values())
    for entry in list(_ENTRIES.values()):
        stale = entry["array"]() is None or entry.get("stale", False)
        if entry["users"] == 0 and (
            stale or len(_ENTRIES) > _MAX_ENTRIES or total > _MAX_BYTES
        ):
            total -= entry["ref"].nbytes
            _drop(entry)


def acquire_array(array):
    """Return a leased snapshot, or None when safe sharing is unavailable.

    A fingerprint detects in-place mutation between phases/fits. The caller
    must not mutate an input concurrently with snapshot creation.
    """
    if (
        not _SHARING_SUPPORTED
        or not isinstance(array, np.ndarray)
        or array.dtype.kind not in "biufc"
        or not array.size
        or array.nbytes > _MAX_BYTES
    ):
        return None
    contiguous = np.ascontiguousarray(array)
    digest = blake2b(memoryview(contiguous).cast("B"), digest_size=16).digest()
    identity = (os.getpid(), id(array))
    with _LOCK:
        entry = _BY_ID.get(identity)
        if (
            entry is not None
            and entry["array"]() is array
            and entry["digest"] == digest
            and entry["ref"].shape == array.shape
            and entry["ref"].dtype == array.dtype.str
        ):
            entry["users"] += 1
            _ENTRIES.move_to_end(entry["ref"].name)
            return entry
        if entry is not None:
            entry["stale"] = True
        _trim()
        total = sum(item["ref"].nbytes for item in _ENTRIES.values())
        for old in list(_ENTRIES.values()):
            if len(_ENTRIES) < _MAX_ENTRIES and total + contiguous.nbytes <= _MAX_BYTES:
                break
            if old["users"] == 0:
                total -= old["ref"].nbytes
                _drop(old)
        if len(_ENTRIES) >= _MAX_ENTRIES or total + contiguous.nbytes > _MAX_BYTES:
            # Active fits retain their leases. New work uses normal argument
            # serialization instead of exceeding the shared dataset budget.
            return None
        try:
            memory = _open_memory(create=True, size=contiguous.nbytes)
        except OSError:
            # Serialization remains a correct portable fallback.
            return None
        try:
            target = np.ndarray(
                contiguous.shape, dtype=contiguous.dtype, buffer=memory.buf
            )
            target[:] = contiguous
            del target
        except BaseException:
            memory.unlink()
            memory.close()
            raise
        ref = SharedArray(memory.name, array.shape, array.dtype.str, array.nbytes)

        def expired(_reference):
            with _LOCK:
                _trim()

        entry = {
            "ref": ref,
            "memory": memory,
            "identity": identity,
            "digest": digest,
            "array": weakref.ref(array, expired),
            "users": 1,
        }
        _ENTRIES[ref.name] = entry
        _BY_ID[identity] = entry
        _trim()
        return entry


def retain_array(entry):
    """Register another user of the shared array."""
    with _LOCK:
        entry["users"] += 1


def release_array(entry):
    """Release a user's reference and trim the shared array cache."""
    with _LOCK:
        entry["users"] -= 1
        _trim()


def resolve_array(ref):
    """Resolve a shared-memory reference to a worker-local array view."""
    if not isinstance(ref, SharedArray):
        return ref
    cached = _WORKER_DATA.pop(ref.name, None)
    if cached is None:
        memory = _open_memory(name=ref.name)
        array = np.ndarray(ref.shape, dtype=np.dtype(ref.dtype), buffer=memory.buf)
        array.flags.writeable = False
        cached = (array, memory)
    _WORKER_DATA[ref.name] = cached
    return cached[0]


def call_with_shared_data(function, arguments):
    """Call the function with resolved array views and keep them available for the call."""
    pinned = {value.name for value in arguments if isinstance(value, SharedArray)}
    resolved = tuple(resolve_array(value) for value in arguments)
    # Never close an array belonging to this invocation, even if the caller
    # returns a view that is serialized immediately after this function returns.
    total = sum(value[0].nbytes for value in _WORKER_DATA.values())
    for token in list(_WORKER_DATA):
        if len(_WORKER_DATA) <= _MAX_ENTRIES and total <= _MAX_BYTES:
            break
        if token not in pinned:
            old_array, old_memory = _WORKER_DATA.pop(token)
            total -= old_array.nbytes
            del old_array
            old_memory.close()
    return function(*resolved)


def initialize_worker():
    """Install process-local attachment lifetime management before first task."""
    atexit.register(clear_worker_data)


def clear_worker_data():
    """Close all shared-memory handles cached by this worker."""
    while _WORKER_DATA:
        _token, (array, memory) = _WORKER_DATA.popitem()
        del array
        memory.close()


def clear_shared_data():
    """Remove shared arrays with no active users from the parent cache."""
    with _LOCK:
        for entry in list(_ENTRIES.values()):
            if entry["users"] == 0:
                _drop(entry)


atexit.register(clear_shared_data)
