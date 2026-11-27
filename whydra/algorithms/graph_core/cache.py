"""Bounded caches scoped to one dataset/test instance or one search phase."""

import weakref
from collections import OrderedDict
from collections.abc import ItemsView, MutableMapping, ValuesView


class _CacheEntry:
    __slots__ = ("key", "members", "references", "value")

    def __init__(self, key, value, slots):
        self.key = key
        self.value = value
        self.members: list[_CacheLink | None] = [None] * slots
        self.references = 0


class _CacheLink:
    __slots__ = ("entry", "next", "previous")

    def __init__(self, entry: _CacheEntry | None = None):
        self.entry = entry
        self.previous = self.next = self


def _unlink(link):
    link.previous.next = link.next
    link.next.previous = link.previous


def _insert_before(link, following):
    link.previous, link.next = following.previous, following
    following.previous.next = link
    following.previous = link


class CIValueStore:
    """One local key/value index, shared by independently ordered cache views.

    The two historical cache layers have observable hit and eviction policies.
    Their membership links retain those policies without two dictionaries of
    probabilities. An entry disappears after its last view releases it.
    """

    def __init__(self, lock):
        self.lock = lock
        self.entries = {}
        self.version = 0
        self._slots = 0
        self._free_slots = []

    def view(self, max_entries=None, *, move_on_write=True):
        """Create a cache view with its own ordering and optional entry limit."""
        with self.lock:
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                slot = self._slots
                self._slots += 1
            return SharedCacheView(self, slot, max_entries, move_on_write)

    def _release(self, link, slot):
        entry = link.entry
        _unlink(link)
        entry.members[slot] = None
        entry.references -= 1
        if not entry.references:
            del self.entries[entry.key]
        self.version += 1


def _release_view(owner, slot, head):
    # No strong owner -> view edge: phase-local outer views can be collected
    # while a caller keeps its CIT alive across fits.
    with owner.lock:
        while head.next is not head:
            owner._release(head.next, slot)
        owner._free_slots.append(slot)


class _SharedItems(ItemsView):
    def __iter__(self):
        for entry in self._mapping._entries():
            yield entry.key, entry.value


class _SharedValues(ValuesView):
    def __iter__(self):
        for entry in self._mapping._entries():
            yield entry.value


class SharedCacheView(MutableMapping):
    """Ordered mapping facade over a single dataset-local CIValueStore.

    CIT owns its capacity and eviction counter; that view has no automatic
    capacity limit. Outer views retain BoundedCache's write-LRU policy.
    Reads do not move an entry: CIT explicitly calls move_to_end on a hit.
    """

    def __init__(self, owner, slot, max_entries, move_on_write):
        self.owner = owner
        self._slot = slot
        self._head = _CacheLink()
        self._size = 0
        self._revision = 0
        self.max_entries = max_entries
        self.evictions = 0
        self._move_on_write = move_on_write
        self._finalizer = weakref.finalize(self, _release_view, owner, slot, self._head)

    def _node(self, key):
        entry = self.owner.entries.get(key)
        return (
            entry.members[self._slot]
            if entry is not None and self._slot < len(entry.members)
            else None
        )

    def contains_entry(self, entry):
        """Test a previously looked-up entry without another hash lookup."""
        return (
            entry is not None
            and self._slot < len(entry.members)
            and entry.members[self._slot] is not None
        )

    def touch_entry(self, entry):
        """Move a known member to the LRU tail without looking up its key."""
        with self.owner.lock:
            if not self.contains_entry(entry):
                raise KeyError(entry.key if entry is not None else None)
            node = entry.members[self._slot]
            if node.next is not self._head:
                _unlink(node)
                _insert_before(node, self._head)
                self._revision += 1

    def __getitem__(self, key):
        node = self._node(key)
        if node is None:
            raise KeyError(key)
        return node.entry.value

    def get(self, key, default=None):
        """Return the cached value for a key, or the supplied default if absent."""
        node = self._node(key)
        return default if node is None else node.entry.value

    def get_and_touch(self, key, default=None):
        """CIT's lookup-and-LRU operation with a single canonical lookup."""
        with self.owner.lock:
            node = self._node(key)
            if node is None:
                return default
            if node.next is not self._head:
                _unlink(node)
                _insert_before(node, self._head)
                self._revision += 1
            return node.entry.value

    def __contains__(self, key):
        return self._node(key) is not None

    def __len__(self):
        return self._size

    def _entries(self, reverse=False):
        head = self._head
        revision = self._revision
        node = head.previous if reverse else head.next
        while node is not head:
            if revision != self._revision:
                msg = "cache changed during iteration"
                raise RuntimeError(msg)
            assert node.entry is not None
            yield node.entry
            node = node.previous if reverse else node.next
        if revision != self._revision:
            msg = "cache changed during iteration"
            raise RuntimeError(msg)

    def __iter__(self):
        for entry in self._entries():
            yield entry.key

    def __reversed__(self):
        for entry in self._entries(reverse=True):
            yield entry.key

    def items(self):
        """Return a view of the key-value pairs in this cache's order."""
        return _SharedItems(self)

    def values(self):
        """Return a view of the cached values in this cache's order."""
        return _SharedValues(self)

    def __setitem__(self, key, value):
        self._set(key, value, self._move_on_write)

    def set_and_touch(self, key, value):
        """Publish a CIT batch result with one lookup and a final LRU touch."""
        self._set(key, value, True)

    def _set(self, key, value, move_on_write):
        owner = self.owner
        with owner.lock:
            entry = owner.entries.get(key)
            if entry is None:
                entry = _CacheEntry(key, value, owner._slots)
                owner.entries[key] = entry
            else:
                entry.value = value
                if len(entry.members) <= self._slot:
                    entry.members.extend([None] * (self._slot + 1 - len(entry.members)))
            node = entry.members[self._slot]
            if node is None:
                node = _CacheLink(entry)
                entry.members[self._slot] = node
                entry.references += 1
                self._size += 1
                self._revision += 1
                _insert_before(node, self._head)
            elif move_on_write and node.next is not self._head:
                _unlink(node)
                _insert_before(node, self._head)
                self._revision += 1
            owner.version += 1
            while self.max_entries is not None and self._size > self.max_entries:
                owner._release(self._head.next, self._slot)
                self._size -= 1
                self._revision += 1
                self.evictions += 1

    def __delitem__(self, key):
        with self.owner.lock:
            node = self._node(key)
            if node is None:
                raise KeyError(key)
            self.owner._release(node, self._slot)
            self._size -= 1
            self._revision += 1

    def move_to_end(self, key, last=True):
        """Move a key to the end of this view, or to the front if last is false."""
        with self.owner.lock:
            node = self._node(key)
            if node is None:
                raise KeyError(key)
            if (node.next is self._head) if last else (node.previous is self._head):
                return
            _unlink(node)
            _insert_before(node, self._head if last else self._head.next)
            self._revision += 1

    def popitem(self, last=True):
        """Remove and return the last item, or the first if last is false."""
        with self.owner.lock:
            if not self._size:
                msg = "dictionary is empty"
                raise KeyError(msg)
            node = self._head.previous if last else self._head.next
            assert node.entry is not None
            result = node.entry.key, node.entry.value
            self.owner._release(node, self._slot)
            self._size -= 1
            self._revision += 1
            return result

    def clear(self):
        """Release this view's entries while preserving entries held by other views."""
        with self.owner.lock:
            while self._head.next is not self._head:
                self.owner._release(self._head.next, self._slot)
            if self._size:
                self._revision += 1
            self._size = 0

    def replace_value(self, key, value):
        """Refine canonical probability without changing either LRU order."""
        with self.owner.lock:
            node = self._node(key)
            if node is None:
                raise KeyError(key)
            node.entry.value = value
            self.owner.version += 1

    def copy(self):
        """Return an independent ordered mapping of this view's current items."""
        with self.owner.lock:
            return OrderedDict(self.items())

    def __reduce__(self):
        # Explicit serialization is a standalone snapshot, never a cross-process
        # proxy. Native search workers receive bounded plain seed dictionaries.
        return (
            _restore_cache_snapshot,
            (self.max_entries, self.evictions, list(self.items())),
        )


def _restore_cache_snapshot(max_entries, evictions, items):
    if max_entries is None:
        return OrderedDict(items)
    cache = BoundedCache(max_entries)
    cache.update(items)
    cache.evictions = evictions
    return cache


def share_ci_cache(cit):
    """Adopt an existing CIT memo without changing membership or counters."""
    with cit._lock:
        if isinstance(cit._cache, SharedCacheView):
            return cit._cache.owner
        owner = CIValueStore(cit._lock)
        primary = owner.view(move_on_write=False)
        primary.update(cit._cache)
        cit._cache = primary
        return owner


class BoundedCache(OrderedDict):
    """Cache entries in write order and count evictions when the limit is exceeded."""

    def __init__(self, max_entries=100000):
        super().__init__()
        self.max_entries = max_entries
        self.evictions = 0

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)
            self.evictions += 1


def ci_cache_for(cit):
    """Choose a shared Fisher-test cache view or a separate bounded cache."""
    from .ci_batch import supports_fisher_batch

    if supports_fisher_batch(cit):
        return share_ci_cache(cit).view(getattr(cit, "cache_size", 100000))
    return BoundedCache(getattr(cit, "cache_size", 100000))
