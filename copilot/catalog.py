"""In-memory catalog index: weighted BM25 + attribute tables + category buckets.

Design notes
------------
The retrieval problem here is unusual: the opening customer message leaks the target's
own coarse category, which collapses a 50,000-product catalog to a median of ~180
candidates. Scoring is therefore dominated by precision within a small pool, not by
fast global recall. That drives two choices:

* A hand-rolled weighted-BM25 inverted index rather than SQLite FTS5, because we need to
  score an arbitrary candidate subset and blend in constraint-satisfaction and prior
  terms that the FTS5 bm25() function cannot express.
* Field weighting folded into the stored term frequency (one postings list carrying a
  weighted tf) instead of six per-field indexes. Same ranking effect, a sixth of the
  memory.

Everything is pure standard library, so the agent runs with no third-party dependency
and no network access, which is what the submission rules require for official scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from array import array
from collections import defaultdict
from pathlib import Path

from copilot.text import (
    COLORS,
    COLOR_RE,
    MATERIAL_RE,
    MATERIALS,
    coarse_category,
    match_text,
    normalise,
    parse_price,
    searchable_text,
    terms,
)

# Field weights folded into the stored term frequency. Ratios follow the starter
# baseline's column weights so the comparison against it stays honest.
FIELD_WEIGHTS: dict[str, int] = {
    "title": 6,
    "categories": 4,
    "features": 3,
    "details": 3,
    "store": 2,
    "description": 1,
}

BM25_K1 = 1.4
BM25_B = 0.72

CACHE_VERSION = 5

_MATERIAL_BIT = {name: 1 << i for i, name in enumerate(MATERIALS)}
_COLOR_BIT = {name: 1 << i for i, name in enumerate(COLORS)}
# gray and grey are the same colour; collapse so a constraint on one matches the other.
_COLOR_BIT["grey"] = _COLOR_BIT["gray"]
#: Canonical colour ordering used by ``first_color`` (grey folded into gray).
_COLOR_CANON = [name for name in COLORS if name != "grey"]


def _field_text(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


class CatalogIndex:
    """Immutable, in-memory view of the frozen product catalog."""

    def __init__(self, path: str | Path, *, use_cache: bool = True) -> None:
        self.path = Path(path)
        cache_file = self._cache_path()
        if use_cache and cache_file is not None and cache_file.exists():
            try:
                self._load_cache(cache_file)
                return
            except Exception:
                # A corrupt or version-skewed cache must never be fatal.
                pass
        self._build()
        if use_cache and cache_file is not None:
            try:
                self._save_cache(cache_file)
            except Exception:
                # Read-only judging environments are expected; caching is best-effort.
                pass

    # ------------------------------------------------------------------ construction
    def _digest(self) -> str:
        hasher = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def _cache_path(self) -> Path | None:
        try:
            digest = self._digest()
        except OSError:
            return None
        return self.path.parent.parent / "artifacts" / f"index-v{CACHE_VERSION}-{digest}.pkl"

    def _build(self) -> None:
        ids: list[str] = []
        blob: list[str] = []
        titles: list[str] = []
        categories: list[str] = []
        prices: list[float] = []
        ratings = array("f")
        rating_counts = array("i")
        material_bits = array("i")
        color_bits = array("i")
        # The simulator discloses the FIRST material/colour it finds, so recording
        # which one that is makes constraint matching and entropy estimates exact.
        first_material = array("b")
        first_color = array("b")

        vocab: dict[str, int] = {}
        raw_postings: dict[int, list[tuple[int, int]]] = defaultdict(list)
        doc_len = array("i")

        with self.path.open(encoding="utf-8") as handle:
            for doc_id, line in enumerate(handle):
                if not line.strip():
                    continue
                product = json.loads(line)
                ids.append(str(product["parent_asin"]))
                titles.append(str(product.get("title") or ""))

                weighted: dict[int, int] = defaultdict(int)
                for field, weight in FIELD_WEIGHTS.items():
                    text = _field_text(product.get(field))
                    if not text:
                        continue
                    for token in terms(text):
                        term_id = vocab.get(token)
                        if term_id is None:
                            term_id = len(vocab)
                            vocab[token] = term_id
                        weighted[term_id] += weight

                total = 0
                for term_id, tf in weighted.items():
                    raw_postings[term_id].append((doc_id, tf))
                    total += tf
                doc_len.append(total)

                text_blob = normalise(searchable_text(product))
                # Stored flattened: phrase matching compares against this directly.
                blob.append(match_text(text_blob))
                categories.append(
                    coarse_category([str(v) for v in (product.get("categories") or [])])
                )

                mask = 0
                first = -1
                for match in MATERIAL_RE.finditer(text_blob):
                    name = match.group(1).lower()
                    mask |= _MATERIAL_BIT[name]
                    if first < 0:
                        first = MATERIALS.index(name)
                material_bits.append(mask)
                first_material.append(first)

                mask = 0
                first = -1
                for match in COLOR_RE.finditer(text_blob):
                    name = match.group(1).lower()
                    mask |= _COLOR_BIT[name]
                    if first < 0:
                        first = _COLOR_CANON.index("gray" if name == "grey" else name)
                color_bits.append(mask)
                first_color.append(first)

                price = parse_price(product.get("price"))
                prices.append(-1.0 if price is None else price)
                try:
                    ratings.append(float(product.get("average_rating") or 0.0))
                except (TypeError, ValueError):
                    ratings.append(0.0)
                try:
                    rating_counts.append(int(product.get("rating_number") or 0))
                except (TypeError, ValueError):
                    rating_counts.append(0)

        self.ids = ids
        self.id_to_doc = {value: index for index, value in enumerate(ids)}
        self.blob = blob
        self.titles = titles
        self.categories = categories
        self.price = array("f", prices)
        self.rating = ratings
        self.rating_count = rating_counts
        self.material_bits = material_bits
        self.color_bits = color_bits
        self.first_material = first_material
        self.first_color = first_color
        self.doc_len = doc_len
        self.vocab = vocab

        # Freeze postings into parallel arrays: compact and cheap to walk.
        self.postings: dict[int, tuple[array, array]] = {}
        for term_id, entries in raw_postings.items():
            docs = array("i", [doc for doc, _ in entries])
            freqs = array("i", [tf for _, tf in entries])
            self.postings[term_id] = (docs, freqs)

        buckets: dict[str, array] = defaultdict(lambda: array("i"))
        for doc_id, name in enumerate(categories):
            buckets[name].append(doc_id)
        self.buckets = dict(buckets)
        # Longest-first, so message parsing can prefer the most specific match.
        self.bucket_names = sorted(self.buckets, key=len, reverse=True)
        self.bucket_lookup = {normalise(name): name for name in self.buckets}

        count = len(ids)
        self.count = count
        self.avg_doc_len = (sum(doc_len) / count) if count else 1.0
        self.idf = {
            term_id: math.log(1.0 + (count - len(docs) + 0.5) / (len(docs) + 0.5))
            for term_id, (docs, _) in self.postings.items()
        }
        max_ratings = max(rating_counts) if count else 1
        self.log_max_ratings = math.log1p(max_ratings) or 1.0

    # ------------------------------------------------------------------------ cache
    _CACHE_ATTRS = (
        "ids", "blob", "titles", "categories", "price", "rating", "rating_count",
        "material_bits", "color_bits", "first_material", "first_color",
        "doc_len", "vocab", "postings", "buckets",
        "bucket_names", "bucket_lookup", "count", "avg_doc_len", "idf",
        "log_max_ratings",
    )

    def _save_cache(self, cache_file: Path) -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: getattr(self, name) for name in self._CACHE_ATTRS}
        payload["__version__"] = CACHE_VERSION
        tmp = cache_file.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_file)

    def _load_cache(self, cache_file: Path) -> None:
        with cache_file.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("__version__") != CACHE_VERSION:
            raise ValueError("cache version mismatch")
        for name in self._CACHE_ATTRS:
            setattr(self, name, payload[name])
        self.id_to_doc = {value: index for index, value in enumerate(self.ids)}

    # --------------------------------------------------------------------- querying
    def bm25(
        self,
        tokens: list[str],
        *,
        candidates: set[int] | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[int, float]:
        """Weighted BM25 over the whole catalog or a restricted candidate subset.

        ``weights`` optionally scales individual query terms, which the retrieval layer
        uses to make confirmed hard constraints outweigh incidental chatter.
        """
        scores: dict[int, float] = defaultdict(float)
        k1, b = BM25_K1, BM25_B
        avg = self.avg_doc_len or 1.0
        doc_len = self.doc_len
        seen: set[int] = set()
        for token in tokens:
            term_id = self.vocab.get(token)
            if term_id is None or term_id in seen:
                continue
            seen.add(term_id)
            entry = self.postings.get(term_id)
            if entry is None:
                continue
            docs, freqs = entry
            idf = self.idf.get(term_id, 0.0)
            boost = 1.0 if weights is None else weights.get(token, 1.0)
            if boost == 0.0:
                continue
            factor = idf * boost
            for position in range(len(docs)):
                doc_id = docs[position]
                if candidates is not None and doc_id not in candidates:
                    continue
                tf = freqs[position]
                norm = 1.0 - b + b * (doc_len[doc_id] / avg)
                scores[doc_id] += factor * (tf * (k1 + 1.0)) / (tf + k1 * norm)
        return scores

    def popularity(self, doc_id: int) -> float:
        """Rating-count prior in [0, 1], damped by log and scaled by average rating."""
        count = math.log1p(self.rating_count[doc_id]) / self.log_max_ratings
        quality = (self.rating[doc_id] or 0.0) / 5.0
        return 0.65 * count + 0.35 * quality

    def bucket(self, category: str) -> array:
        return self.buckets.get(category, array("i"))
