"""Tests for the index cache and its restricted unpickler.

`pickle.load` on a file an attacker can write is arbitrary code execution, and the index
cache lives in `artifacts/`. The loader is therefore restricted to the types the cache
actually contains.

The restriction has a failure mode that is worse than an exception, and these tests exist
because of it: a rejected cache is treated as a cache *miss*, so an allowlist that is too
narrow does not raise -- it silently rebuilds the index on every single start. That is a
50x startup regression with no error anywhere, and it is exactly what happened when the
allowlist first omitted `array._array_reconstructor`. A round trip is asserted here so the
next person to touch the allowlist finds out immediately.
"""

from __future__ import annotations

import io
import os
import pathlib
import pickle
from array import array

import pytest

from copilot.catalog import _ALLOWED_PICKLE_TYPES, CatalogIndex, _RestrictedUnpickler


def _round_trip(value: object) -> object:
    blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    return _RestrictedUnpickler(io.BytesIO(blob)).load()


# ------------------------------------------------------------- the cache must survive
def test_every_type_the_cache_holds_survives_the_round_trip():
    """A payload shaped like `_CACHE_ATTRS`: arrays, lists, dicts, and scalars."""
    payload = {
        "ids": ["B001", "B002"],
        "rating": array("f", [4.5, 3.0]),
        "rating_count": array("i", [10, 20]),
        "postings": {"leather": [0, 1]},
        "avg_doc_len": 42.5,
        "count": 2,
        "bucket_names": ("shoes", "bags"),
        "__version__": 6,
    }
    restored = _round_trip(payload)
    assert restored == payload
    assert isinstance(restored["rating"], array)
    assert restored["rating"].typecode == "f"


@pytest.mark.parametrize(
    "value",
    [
        array("f", [1.5, 2.5]),
        array("i", [1, 2, 3]),
        array("d", [1.0]),
        {"a": [1, 2], "b": (3, 4)},
        frozenset({1, 2}),
        {"nested": {"deep": [array("i", [7])]}},
    ],
)
def test_individual_cache_types_round_trip(value):
    assert _round_trip(value) == value


# ------------------------------------------------------------------ the restriction
def test_a_code_execution_payload_is_refused():
    """The whole reason the restricted unpickler exists."""

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    blob = pickle.dumps(Evil())
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        _RestrictedUnpickler(io.BytesIO(blob)).load()


def test_an_arbitrary_class_is_refused():
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        _RestrictedUnpickler(io.BytesIO(pickle.dumps(pytest.approx(1.0)))).load()


def test_the_allowlist_names_no_callable_that_executes_anything():
    """Nothing in the allowlist may be a module that can run a command or import code."""
    forbidden_modules = {"os", "nt", "posix", "subprocess", "builtins.eval", "importlib"}
    for module, name in _ALLOWED_PICKLE_TYPES:
        assert module not in forbidden_modules, f"{module}.{name} is dangerous"
        assert name not in {"eval", "exec", "system", "popen", "__import__"}


def test_the_allowlist_is_restricted_to_data_types():
    assert all(module in {"array", "builtins"} for module, _ in _ALLOWED_PICKLE_TYPES)


# ------------------------------------------------------- the end-to-end regression test
def test_a_real_index_survives_a_save_and_reload(synthetic_catalog, monkeypatch, tmp_path):
    """Build, cache, reload -- the path that silently degraded when the allowlist was wrong.

    The unit tests above pass a hand-written payload, which is exactly why they did not
    catch the original bug: it only appears once a genuine `CatalogIndex` is pickled, since
    that is what puts `array` objects through `_array_reconstructor`. This builds one.
    """
    monkeypatch.setattr(
        CatalogIndex, "_cache_path", lambda self: tmp_path / "index.pkl", raising=True
    )

    built = CatalogIndex(synthetic_catalog)
    assert (tmp_path / "index.pkl").exists(), "the build should have written a cache"

    reloaded = CatalogIndex.__new__(CatalogIndex)
    reloaded.path = pathlib.Path(str(synthetic_catalog))
    reloaded._load_cache(tmp_path / "index.pkl")   # must not raise

    assert reloaded.count == built.count
    assert reloaded.ids == built.ids
    assert list(reloaded.rating) == list(built.rating)
    assert reloaded.id_to_doc == built.id_to_doc


def test_a_cache_the_loader_rejects_is_treated_as_a_miss_not_a_crash(
    synthetic_catalog, monkeypatch, tmp_path
):
    """A hostile or unreadable cache must degrade to a rebuild, never take the agent down."""
    cache = tmp_path / "index.pkl"
    monkeypatch.setattr(CatalogIndex, "_cache_path", lambda self: cache, raising=True)
    CatalogIndex(synthetic_catalog)

    cache.write_bytes(pickle.dumps({"__version__": 6, "evil": pytest.approx(1.0)}))
    rebuilt = CatalogIndex(synthetic_catalog)          # must not raise
    assert rebuilt.count > 0
