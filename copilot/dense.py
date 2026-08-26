"""Dense retrieval at inference time, on the standard library alone.

The artifact this reads is built by `tools/build_vectors.py`, which needs numpy and scipy.
Nothing here does. That split is the point: the scientific stack is a *build* dependency,
so the shipped agent keeps its dependency-free property and still gets vector similarity.

Two facts make a pure-Python implementation viable, and both are properties of this problem
rather than clever coding:

* **The category lock has already done the hard part.** Scoring happens over a pool of
  roughly 180 candidates, not 50,000, so this decodes ~180 rows per turn rather than the
  whole matrix. Measured at ~6.7 ms per turn for a 128-dimensional space -- real, but
  affordable against a 37 ms budget.
* **`struct` speaks IEEE-754 half precision.** The `'e'` format code means fp16 rows can be
  unpacked directly out of an mmap, so the artifact is half the size of fp32 and no
  conversion layer is needed.

Document rows are L2-normalised at build time, so cosine similarity is a plain dot product
here and no norm is ever computed at inference.

Everything degrades to nothing. A missing artifact, a truncated file, a version bump, or a
catalog whose digest does not match the one the vectors were built from all leave
``available()`` false, and the agent then ranks exactly as it does without this module. A
stale artifact silently paired with a different catalog would be far worse than no artifact
at all, which is why the digest is checked rather than assumed.
"""

from __future__ import annotations

import json
import math
import mmap
import struct
from pathlib import Path

#: Bumped when the artifact layout changes, so an old directory is refused rather than
#: misread. Version skew in a binary format is silent otherwise.
ARTIFACT_VERSION = 1

DOCS_FILE = "docs.f16"
TERMS_FILE = "terms.f16"
TERMS_LIST = "terms.txt"
META_FILE = "meta.json"

_BYTES_PER_VALUE = 2


class DenseVectors:
    """Read-only view over the latent-space artifact.

    Construction never raises. Callers check :meth:`available` and carry on regardless.
    """

    def __init__(self, path: str | Path, *, catalog_digest: str | None = None) -> None:
        self.path = Path(path)
        self.dim = 0
        self.kind = ""
        self.meta: dict = {}
        self._docs: mmap.mmap | None = None
        self._terms: mmap.mmap | None = None
        self._handles: list = []
        self._term_row: dict[str, int] = {}
        self._unpack = None
        self._doc_count = 0
        self._reason = "not loaded"
        try:
            self._load(catalog_digest)
        except Exception as error:  # pragma: no cover - defensive
            self.close()
            self._reason = f"{type(error).__name__}: {error}"

    # ------------------------------------------------------------------------ loading
    def _load(self, catalog_digest: str | None) -> None:
        meta_path = self.path / META_FILE
        if not meta_path.exists():
            self._reason = f"no artifact at {self.path}"
            return

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != ARTIFACT_VERSION:
            self._reason = (
                f"artifact version {meta.get('version')!r}, expected {ARTIFACT_VERSION}"
            )
            return
        if catalog_digest is not None and meta.get("catalog_digest") != catalog_digest:
            # The vectors describe a different catalog. Using them would silently score
            # against the wrong products, which is worse than not scoring at all.
            self._reason = "artifact was built from a different catalog"
            return

        dim = int(meta.get("dim") or 0)
        if dim <= 0:
            self._reason = f"invalid dimension {dim!r}"
            return

        terms = (self.path / TERMS_LIST).read_text(encoding="utf-8").split("\n")
        terms = [term for term in terms if term]

        docs_map = self._map(self.path / DOCS_FILE)
        terms_map = self._map(self.path / TERMS_FILE)
        row_bytes = dim * _BYTES_PER_VALUE

        if len(terms_map) != len(terms) * row_bytes:
            self._reason = "terms.f16 does not match terms.txt"
            self.close()
            return
        if len(docs_map) % row_bytes:
            self._reason = "docs.f16 is truncated"
            self.close()
            return

        self.meta = meta
        self.dim = dim
        self.kind = str(meta.get("kind") or "unknown")
        self._docs = docs_map
        self._terms = terms_map
        self._term_row = {term: row for row, term in enumerate(terms)}
        self._unpack = struct.Struct(f"<{dim}e").unpack_from
        self._doc_count = len(docs_map) // row_bytes
        self._reason = ""

    def _map(self, path: Path) -> mmap.mmap:
        handle = path.open("rb")
        self._handles.append(handle)
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._handles.append(mapped)
        return mapped

    def close(self) -> None:
        for handle in reversed(self._handles):
            try:
                handle.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._handles.clear()
        self._docs = None
        self._terms = None

    # ------------------------------------------------------------------------- status
    def available(self) -> bool:
        return self._docs is not None and self._terms is not None and self.dim > 0

    @property
    def reason(self) -> str:
        """Why the artifact is unusable, for diagnostics. Empty when it loaded."""
        return self._reason

    def __len__(self) -> int:
        return self._doc_count

    # ------------------------------------------------------------------------ queries
    def query_vector(
        self, tokens: list[str], weights: dict[str, float] | None = None
    ) -> tuple[float, ...] | None:
        """Project a bag of query terms into the latent space.

        The document side is ``X V``; a query projects the same way, and because a query is
        a sparse term vector that reduces to a weighted sum of the rows of the term matrix.
        Unknown terms contribute nothing -- they are dropped from the artifact precisely
        because they had no co-occurrence structure to learn from, and BM25 scores them
        better anyway.
        """
        if not self.available() or not tokens:
            return None

        total = [0.0] * self.dim
        seen: set[str] = set()
        hits = 0
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            row = self._term_row.get(token)
            if row is None:
                continue
            weight = 1.0 if weights is None else weights.get(token, 1.0)
            if weight == 0.0:
                continue
            vector = self._unpack(self._terms, row * self.dim * _BYTES_PER_VALUE)
            for position in range(self.dim):
                total[position] += weight * vector[position]
            hits += 1

        if not hits:
            return None
        norm = math.sqrt(sum(value * value for value in total))
        if norm <= 0.0:
            return None
        return tuple(value / norm for value in total)

    def similarity(self, query: tuple[float, ...], doc_id: int) -> float:
        """Cosine similarity. Both sides are unit vectors, so this is a dot product."""
        if not self.available() or not 0 <= doc_id < self._doc_count:
            return 0.0
        row = self._unpack(self._docs, doc_id * self.dim * _BYTES_PER_VALUE)
        return sum(map(float.__mul__, query, row))
