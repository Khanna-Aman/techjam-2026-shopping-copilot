"""Tests for the offline dense retrieval tier.

The artifact is a pair of raw fp16 blobs bound to a catalog by digest. Almost everything
that can go wrong with that is silent: a stale artifact paired with a different catalog
would score against the wrong products and still return plausible numbers. So most of these
tests are about refusing to load, not about loading.

No numpy here, and none in `copilot/dense.py` either -- the fixtures below write the binary
format with `struct`, exactly as the module reads it.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from copilot.agent import ShoppingCopilot
from copilot.config import DEFAULT_CONFIG, AgentConfig
from copilot.dense import (
    ARTIFACT_VERSION,
    DOCS_FILE,
    META_FILE,
    TERMS_FILE,
    TERMS_LIST,
    DenseVectors,
)

DIM = 4
DIGEST = "abc123"


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _write(directory: Path, *, docs=None, terms=None, dim=DIM, version=ARTIFACT_VERSION,
           digest=DIGEST, term_names=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    docs = docs if docs is not None else [_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]),
                                          _unit([1, 1, 0, 0])]
    terms = terms if terms is not None else [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    term_names = term_names if term_names is not None else ["alpha", "beta", "gamma"]

    packer = struct.Struct(f"<{dim}e").pack
    (directory / DOCS_FILE).write_bytes(b"".join(packer(*row) for row in docs))
    (directory / TERMS_FILE).write_bytes(b"".join(packer(*row) for row in terms))
    (directory / TERMS_LIST).write_text("\n".join(term_names) + "\n", encoding="utf-8")
    (directory / META_FILE).write_text(
        json.dumps({"version": version, "kind": "lsa", "dim": dim,
                    "docs": len(docs), "terms": len(term_names),
                    "catalog_digest": digest}),
        encoding="utf-8",
    )
    return directory


# ------------------------------------------------------------------- refusing to load
def test_missing_artifact_is_unavailable_not_an_error(tmp_path):
    vectors = DenseVectors(tmp_path / "nope")
    assert vectors.available() is False
    assert "no artifact" in vectors.reason


def test_digest_mismatch_refuses_to_load(tmp_path):
    """The failure this exists to prevent: right vectors, wrong catalog."""
    path = _write(tmp_path / "d", digest="aaa")
    vectors = DenseVectors(path, catalog_digest="bbb")
    assert vectors.available() is False
    assert "different catalog" in vectors.reason


def test_matching_digest_loads(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    assert vectors.available() is True
    assert vectors.reason == ""
    assert vectors.dim == DIM
    assert len(vectors) == 3


def test_absent_digest_check_still_loads(tmp_path):
    """Callers that genuinely have no digest can still opt in."""
    assert DenseVectors(_write(tmp_path / "d")).available() is True


def test_version_skew_refuses_to_load(tmp_path):
    path = _write(tmp_path / "d", version=ARTIFACT_VERSION + 1)
    vectors = DenseVectors(path, catalog_digest=DIGEST)
    assert vectors.available() is False
    assert "version" in vectors.reason


def test_term_matrix_disagreeing_with_term_list_refuses(tmp_path):
    path = _write(tmp_path / "d", term_names=["alpha", "beta", "gamma", "delta"])
    vectors = DenseVectors(path, catalog_digest=DIGEST)
    assert vectors.available() is False
    assert "terms.f16" in vectors.reason


def test_truncated_document_matrix_refuses(tmp_path):
    path = _write(tmp_path / "d")
    blob = (path / DOCS_FILE).read_bytes()
    (path / DOCS_FILE).write_bytes(blob[:-3])
    vectors = DenseVectors(path, catalog_digest=DIGEST)
    assert vectors.available() is False
    assert "truncated" in vectors.reason


def test_corrupt_metadata_is_survivable(tmp_path):
    path = _write(tmp_path / "d")
    (path / META_FILE).write_text("{not json", encoding="utf-8")
    vectors = DenseVectors(path, catalog_digest=DIGEST)
    assert vectors.available() is False


def test_invalid_dimension_refuses(tmp_path):
    path = _write(tmp_path / "d")
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    meta["dim"] = 0
    (path / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    assert DenseVectors(path, catalog_digest=DIGEST).available() is False


# ------------------------------------------------------------------------- projection
def test_query_vector_is_a_unit_vector(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    q = vectors.query_vector(["alpha", "beta"])
    assert q is not None
    assert math.isclose(math.sqrt(sum(v * v for v in q)), 1.0, rel_tol=1e-3)


def test_unknown_terms_contribute_nothing(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    assert vectors.query_vector(["not_in_vocabulary"]) is None
    assert vectors.query_vector(["alpha", "not_in_vocabulary"]) == vectors.query_vector(["alpha"])


def test_empty_and_zero_weighted_queries_return_none(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    assert vectors.query_vector([]) is None
    assert vectors.query_vector(["alpha"], {"alpha": 0.0}) is None


def test_repeated_terms_are_counted_once(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    assert vectors.query_vector(["alpha", "alpha", "alpha"]) == vectors.query_vector(["alpha"])


def test_weights_steer_the_projection(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    mostly_alpha = vectors.query_vector(["alpha", "beta"], {"alpha": 10.0})
    assert mostly_alpha[0] > mostly_alpha[1]


def test_unavailable_vectors_project_to_nothing(tmp_path):
    vectors = DenseVectors(tmp_path / "nope")
    assert vectors.query_vector(["alpha"]) is None
    assert vectors.similarity((1.0, 0.0, 0.0, 0.0), 0) == 0.0


# ------------------------------------------------------------------------- similarity
def test_similarity_recovers_the_aligned_document(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    q = vectors.query_vector(["alpha"])
    assert vectors.similarity(q, 0) == pytest.approx(1.0, abs=1e-3)   # exactly aligned
    assert vectors.similarity(q, 1) == pytest.approx(0.0, abs=1e-3)   # orthogonal
    assert vectors.similarity(q, 0) > vectors.similarity(q, 2) > vectors.similarity(q, 1)


def test_similarity_stays_within_the_cosine_range(tmp_path):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    q = vectors.query_vector(["alpha", "beta"])
    for doc in range(len(vectors)):
        assert -1.001 <= vectors.similarity(q, doc) <= 1.001


@pytest.mark.parametrize("doc_id", [-1, 3, 10_000])
def test_out_of_range_documents_score_zero(tmp_path, doc_id):
    vectors = DenseVectors(_write(tmp_path / "d"), catalog_digest=DIGEST)
    assert vectors.similarity(vectors.query_vector(["alpha"]), doc_id) == 0.0


# ------------------------------------------------------------------ agent integration
def test_dense_is_not_loaded_when_disabled(synthetic_index):
    """Disabled means the catalog is never hashed and no file is opened."""
    assert DEFAULT_CONFIG.use_dense_rerank is False
    agent = ShoppingCopilot("unused", index=synthetic_index)
    assert agent.dense is None


def test_a_missing_artifact_leaves_ranking_unchanged(synthetic_catalog, tmp_path):
    """The load-bearing degradation: enabling the flag without vectors changes nothing."""
    opener = "I'm looking for T-Shirts, but I'm still exploring."
    plain = ShoppingCopilot(synthetic_catalog)
    plain.reset("s", {})
    expected = plain.respond("s", opener, 1, 10)["recommendations"]

    config = AgentConfig(use_dense_rerank=True, dense_path=str(tmp_path / "absent"))
    degraded = ShoppingCopilot(synthetic_catalog, config=config)
    assert degraded.dense is not None
    assert degraded.dense.available() is False
    degraded.reset("s", {})
    assert degraded.respond("s", opener, 1, 10)["recommendations"] == expected
