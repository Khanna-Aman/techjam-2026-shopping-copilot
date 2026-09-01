# Building the dense artifact on a Kaggle GPU

The dense retrieval artifact can be built on CPU — `all-MiniLM-L6-v2` over 50,000 products
takes about 14 minutes on 16 threads. A GPU is worth it for two things: encoding with a
model large enough that "you only tried the small one" stops being an available objection,
and training the cross-encoder, which is real gradient work rather than inference.

Nothing here is a separate implementation. The notebook clones this repository and runs
`tools/build_vectors.py`, the same script used locally, so the GPU path and the CPU path
cannot silently diverge.

## Setup

Create a Kaggle notebook, set **Accelerator → GPU T4 x2** (or P100), and leave internet
**on** — the catalog and the model weights are both fetched.

## Cell 1 — repository and catalog

```python
!git clone -q https://github.com/Khanna-Aman/techjam-2026-shopping-copilot repo
%cd repo

# The frozen catalog, from the organiser's own public release. Not redistributed here.
BASE = "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit"
!curl -fsSL -o catalog.jsonl.gz {BASE}/catalog.jsonl.gz
!curl -fsSL -o SHA256SUMS      {BASE}/SHA256SUMS
!sha256sum -c SHA256SUMS --ignore-missing
!gzip -dc catalog.jsonl.gz > data/catalog.jsonl
!wc -l data/catalog.jsonl        # expect 50000
```

## Cell 2 — dependencies

```python
!pip install -q sentence-transformers
import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Cell 3 — encode

`--batch-size` is the only flag that wants changing on a GPU. The script prints the device
it actually used, so a run that silently fell back to CPU is visible rather than merely slow.

```python
# ~8 minutes on a T4 for a base-sized model.
!python -m tools.build_vectors --mode transformer \
    --model BAAI/bge-base-en-v1.5 \
    --batch-size 512 \
    --out artifacts/dense-bge

!du -sh artifacts/dense-bge && cat artifacts/dense-bge/meta.json
```

Model choices, in rising order of cost and of how thoroughly they close off the "too small"
objection:

| model | dim | docs artifact | note |
|---|---:|---:|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 38 MB | the CPU baseline, already measured |
| `BAAI/bge-base-en-v1.5` | 768 | 77 MB | strong general retriever, good default |
| `intfloat/e5-large-v2` | 1024 | 102 MB | larger; needs `query: ` / `passage: ` prefixes to perform properly |

## Cell 4 — get the artifact out

```python
import shutil
shutil.make_archive("/kaggle/working/dense-bge", "zip", "artifacts/dense-bge")
print("download /kaggle/working/dense-bge.zip from the notebook's Output tab")
```

Unzip it locally into `artifacts/dense-bge/`, then measure exactly as the CPU artifact was
measured:

```bash
python -m tools.sweep --mode dense --base '{"dense_path": "artifacts/dense-bge"}'
```

## What to expect

The LSA tier lost at every weight, and the reason looks structural rather than a property of
the encoder: 94.5% of the constraint strings the simulator discloses appear **verbatim** in
their own target product, and 22.4% are unique to a single product in 50,000 — rising to
99.2% and 29.7% on the 602 of 800 the simulator mines out of the target's own fields rather
than synthesises (`python -m tools.constraint_stats`). Dense retrieval closes vocabulary
mismatch, and this benchmark has very little of it by construction.

A better encoder is therefore expected to lose too. It is worth running anyway, because
"a stronger model was tried and the sign did not change" is a materially different claim
from "a weak model was tried", and the second one invites an obvious question that the first
one answers.
