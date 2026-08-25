"""Shared test fixtures.

Unit tests run against a small synthetic catalog so they stay fast and their assertions
stay readable. The handful of tests that genuinely need catalog scale use the real index,
which is session-scoped and cached so it is built at most once per run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from copilot.catalog import CatalogIndex  # noqa: E402

#: A deliberately small catalog with hand-chosen attributes, so every test can reason
#: about exactly which product should win and why.
SYNTHETIC_PRODUCTS: list[dict] = [
    {
        "parent_asin": "B000000001",
        "title": "Classic Cotton Crew Neck T-Shirt, Black",
        "features": ["100% cotton", "Machine washable", "Ribbed crew neckline"],
        "description": ["A soft everyday tee."],
        "price": 19.99,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts", "T-Shirts"],
        "details": {"Department": "Mens", "Fit Type": "Regular"},
        "average_rating": 4.5,
        "rating_number": 1200,
        "store": "Basics Co",
    },
    {
        "parent_asin": "B000000002",
        "title": "Blue Polyester Running Shirt",
        "features": ["Moisture wicking polyester", "Reflective trim for running"],
        "description": ["Built for the gym."],
        "price": 34.50,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts", "T-Shirts"],
        "details": {"Department": "Mens", "Fit Type": "Athletic"},
        "average_rating": 4.1,
        "rating_number": 300,
        "store": "RunFast",
    },
    {
        "parent_asin": "B000000003",
        "title": "Leather Ankle Boot, Brown",
        "features": ["Genuine leather upper", "Rubber outsole for hiking"],
        "description": ["Durable winter boot."],
        "price": 129.00,
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Boots"],
        "details": {"Department": "Womens"},
        "average_rating": 4.8,
        "rating_number": 90,
        "store": "Wanderer",
    },
    {
        "parent_asin": "B000000004",
        "title": "Silk Scarf, Purple Paisley",
        "features": ["Pure silk", "Hand rolled edges"],
        "description": [],
        "price": None,
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Accessories", "Scarves"],
        "details": {},
        "average_rating": 3.9,
        "rating_number": 12,
        "store": "Maison",
    },
    {
        "parent_asin": "B000000005",
        "title": "Wool Blend Overcoat, Grey",
        "features": ["Wool blend shell", "Fully lined"],
        "description": ["Warm winter coat."],
        "price": 210.00,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Coats", "Overcoats"],
        "details": {"Department": "Mens"},
        "average_rating": 4.3,
        "rating_number": 450,
        "store": "Northwind",
    },
]


@pytest.fixture(scope="session")
def synthetic_catalog(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("catalog") / "catalog.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for product in SYNTHETIC_PRODUCTS:
            handle.write(json.dumps(product) + "\n")
    return path


@pytest.fixture(scope="session")
def synthetic_index(synthetic_catalog: Path) -> CatalogIndex:
    # Caching is disabled so the fixture never collides with the real catalog's cache.
    return CatalogIndex(synthetic_catalog, use_cache=False)


@pytest.fixture(scope="session")
def real_catalog() -> Path:
    path = _ROOT / "data" / "catalog.jsonl"
    if not path.exists():
        pytest.skip("data/catalog.jsonl not present; see README for the download step")
    return path


@pytest.fixture(scope="session")
def real_index(real_catalog: Path) -> CatalogIndex:
    return CatalogIndex(real_catalog)
