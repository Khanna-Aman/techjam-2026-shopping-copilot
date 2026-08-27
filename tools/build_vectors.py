"""Build the dense retrieval artifact. Offline, and never imported by the agent.

This is the only part of the project that uses numpy and scipy. It runs once, produces a
pair of fp16 arrays, and is never touched again -- inference reads those arrays with
`struct` and `mmap` from the standard library alone (see `copilot/dense.py`). The agent's
dependency-free property is therefore preserved: the *build* needs scientific Python, the
*shipped system* does not.

Why latent semantic analysis rather than a neural encoder, for this first tier: it is
derived entirely from the frozen catalog by a script anyone can re-run, it introduces no
third-party model weights or licence question, and -- the part that actually matters -- the
query side and the document side are projections through the *same* matrix, so they are
consistent by construction rather than by approximation. A transformer bi-encoder can score
better, but distilling its query side into a static table is an approximation whose error is
invisible until it costs you something. This tier is the floor, and the floor should be
boring.

Method
------
Build the sparse document-term matrix X using the *same* BM25 term weights the lexical
ranker uses, so the latent space reflects the weighting the rest of the system already
trusts. Take a truncated SVD, X ~ U S V^T, and keep:

    docs.f16    row d = (U S)[d], L2-normalised   -- the document embedding
    terms.f16   row t = V[t]                      -- the projection of one term

A query is a sparse term vector q; its projection into the same space is q V, which is just
the weighted sum of the rows of `terms.f16` for the terms it contains. Cosine similarity is
then a dot product of two normalised vectors. That is the whole of the inference-time maths,
which is why it fits in the standard library.

Terms appearing in a single document are dropped. They nearly triple the term matrix, and
their latent representation is degenerate anyway -- a term with one posting has no
co-occurrence structure to learn from. BM25 already handles them optimally, with a high idf
and a single posting to walk.

Usage:
    pip install numpy scipy scikit-learn
    python -m tools.build_vectors                    # k=128, df>=2
    python -m tools.build_vectors --dim 96 --min-df 3
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from copilot.catalog import BM25_B, BM25_K1, CatalogIndex  # noqa: E402
from copilot.dense import ARTIFACT_VERSION, DOCS_FILE, META_FILE, TERMS_FILE, TERMS_LIST  # noqa: E402


def _require_scientific_python():
    try:
        import numpy  # noqa: PLC0415
        import scipy.sparse  # noqa: PLC0415
        from sklearn.decomposition import TruncatedSVD  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"tools/build_vectors.py needs numpy, scipy and scikit-learn ({error}).\n"
            "  pip install numpy scipy scikit-learn\n\n"
            "Only this build script needs them. The agent reads the artifact it produces\n"
            "with the standard library alone and never imports them."
        ) from error
    return numpy, scipy.sparse, TruncatedSVD


def build_matrix(index: CatalogIndex, min_df: int, np, sparse):
    """The document-term matrix, weighted exactly as `CatalogIndex.bm25` weights it."""
    kept = [
        (term_id, term)
        for term, term_id in index.vocab.items()
        if len(index.postings[term_id][0]) >= min_df
    ]
    kept.sort(key=lambda pair: pair[1])
    column_of = {term_id: column for column, (term_id, _) in enumerate(kept)}
    terms = [term for _, term in kept]

    average = index.avg_doc_len or 1.0
    doc_len = index.doc_len

    rows, cols, values = [], [], []
    for term_id, column in column_of.items():
        docs, freqs = index.postings[term_id]
        idf = index.idf.get(term_id, 0.0)
        if idf <= 0.0:
            continue
        for position in range(len(docs)):
            doc_id = docs[position]
            tf = freqs[position]
            norm = 1.0 - BM25_B + BM25_B * (doc_len[doc_id] / average)
            rows.append(doc_id)
            cols.append(column)
            values.append(idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * norm))

    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(index.count, len(terms)),
    )
    return matrix, terms


def _write_fp16(path: Path, matrix, dim: int) -> None:
    """Write rows as little-endian IEEE-754 half floats, which `struct` can read back."""
    packer = struct.Struct(f"<{dim}e").pack
    with path.open("wb") as handle:
        for row in matrix:
            # Half-precision saturates at 65504; the values here are far smaller, but clip
            # rather than emit an infinity that would poison every later dot product.
            handle.write(packer(*[max(-65000.0, min(65000.0, float(v))) for v in row]))


def _load_encoder(model_name: str):
    try:
        import os  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"--mode transformer needs sentence-transformers ({error}).\n"
            "  pip install sentence-transformers\n\n"
            "Build-time only. The artifact it produces is read by the standard library."
        ) from error
    torch.set_num_threads(os.cpu_count() or 4)
    encoder = SentenceTransformer(model_name)
    encoder.max_seq_length = 128
    device = getattr(encoder, "device", "unknown")
    print(f"encoder    : {model_name} on {device}")
    if str(device).startswith("cpu") and torch.cuda.is_available():  # pragma: no cover
        print("             (CUDA is available but unused -- check the runtime)")
    return encoder


def build_transformer(index: CatalogIndex, catalog: str, model_name: str, min_df: int,
                      np, batch_size: int = 64):
    """Encode products and vocabulary terms with a sentence bi-encoder.

    The query side is *distilled*: each vocabulary term is encoded on its own, and a query
    becomes the weighted mean of its terms' vectors. That keeps inference in the standard
    library, at the cost of being an approximation of what the real encoder would produce
    for a whole sentence. The approximation is measured rather than assumed -- see the
    live-encoder comparison in tools/dense_probe.py.
    """
    from evaluator.local_evaluator import searchable_text  # noqa: PLC0415

    encoder = _load_encoder(model_name)

    texts: list[str] = []
    with Path(catalog).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                texts.append(searchable_text(json.loads(line))[:600])
    print(f"encoding   : {len(texts):,} products with {model_name}")
    docs = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )

    terms = sorted(
        term
        for term, term_id in index.vocab.items()
        if len(index.postings[term_id][0]) >= min_df
    )
    print(f"encoding   : {len(terms):,} vocabulary terms for the distilled query side")
    term_vectors = encoder.encode(
        terms, batch_size=batch_size * 4, show_progress_bar=True, normalize_embeddings=True
    )
    return np.asarray(docs), np.asarray(term_vectors), terms


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dense retrieval artifact")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out", default="artifacts/dense")
    parser.add_argument("--mode", default="lsa", choices=("lsa", "transformer"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="encoder batch size; raise it substantially on a GPU",
    )
    args = parser.parse_args()

    np, sparse, TruncatedSVD = _require_scientific_python()

    started = time.perf_counter()
    index = CatalogIndex(args.catalog)
    print(f"index      : {index.count:,} docs, {len(index.vocab):,} terms")

    explained = None
    if args.mode == "transformer":
        docs, term_vectors, terms = build_transformer(
            index, args.catalog, args.model, args.min_df, np, args.batch_size
        )
        dim = docs.shape[1]
    else:
        matrix, terms = build_matrix(index, args.min_df, np, sparse)
        print(f"matrix     : {matrix.shape[0]:,} x {matrix.shape[1]:,}, {matrix.nnz:,} nonzero")

        svd = TruncatedSVD(
            n_components=args.dim, algorithm="randomized", random_state=args.seed
        )
        docs = svd.fit_transform(matrix)          # (docs, k) = U S
        term_vectors = svd.components_.T          # (terms, k) = V
        explained = float(svd.explained_variance_ratio_.sum())
        print(f"svd        : k={args.dim}, explains {explained:.1%} of variance")
        dim = args.dim

    # Normalise document rows once, at build time, so cosine similarity becomes a plain dot
    # product at inference and the agent never has to compute a norm.
    norms = np.linalg.norm(docs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    docs = docs / norms
    args.dim = dim

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_fp16(out / DOCS_FILE, docs, args.dim)
    _write_fp16(out / TERMS_FILE, term_vectors, args.dim)
    (out / TERMS_LIST).write_text("\n".join(terms) + "\n", encoding="utf-8")
    (out / META_FILE).write_text(
        json.dumps(
            {
                "version": ARTIFACT_VERSION,
                "kind": args.mode,
                "model": args.model if args.mode == "transformer" else None,
                "dim": args.dim,
                "docs": index.count,
                "terms": len(terms),
                "min_df": args.min_df,
                "catalog_digest": index._digest(),
                "explained_variance_ratio": round(explained, 6) if explained is not None else None,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    size = sum(f.stat().st_size for f in out.iterdir()) / 1048576
    print(f"written    : {out} ({size:.1f} MB) in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
