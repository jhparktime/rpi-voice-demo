"""Anchors-based LOCAL/CLOUD router using dataset/anchors.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


MARGIN = float(os.environ.get("ROUTER_MARGIN", "0.03"))
EMBEDDING_MODEL = os.environ.get("ROUTER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@dataclass
class RouteResult:
    mode: str  # "LOCAL" or "CLOUD"
    confidence: float
    matched_anchor: str
    best_local: float
    best_cloud: float
    delta: float


_initialized = False
_embedder: Optional[SentenceTransformer] = None
_local_texts: List[str] = []
_cloud_texts: List[str] = []
_local_emb: Optional[np.ndarray] = None
_cloud_emb: Optional[np.ndarray] = None


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _init_router() -> None:
    """Lazy-load anchors.json and compute embeddings for LOCAL/CLOUD anchors."""
    global _initialized, _embedder, _local_texts, _cloud_texts, _local_emb, _cloud_emb
    if _initialized:
        return

    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "dataset"
    anchors_path = dataset_dir / "anchors.json"
    if not anchors_path.exists():
        raise FileNotFoundError(f"anchors.json not found at {anchors_path}")

    with anchors_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _local_texts = list(data.get("local_anchors", []))
    _cloud_texts = list(data.get("cloud_anchors", []))
    if not _local_texts or not _cloud_texts:
        raise ValueError("anchors.json must contain non-empty local_anchors and cloud_anchors.")

    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    local_emb = _embedder.encode(_local_texts, convert_to_numpy=True).astype(np.float32, copy=False)
    cloud_emb = _embedder.encode(_cloud_texts, convert_to_numpy=True).astype(np.float32, copy=False)
    _local_emb = _normalize_rows(local_emb)
    _cloud_emb = _normalize_rows(cloud_emb)

    _initialized = True


def route_local_or_cloud(text: str) -> RouteResult:
    """Route user text to LOCAL or CLOUD using anchors-based cosine similarity."""
    if not text or not text.strip():
        # Empty input; default to LOCAL with neutral scores.
        return RouteResult(
            mode="LOCAL",
            confidence=0.0,
            matched_anchor="",
            best_local=0.0,
            best_cloud=0.0,
            delta=0.0,
        )

    _init_router()
    assert _embedder is not None
    assert _local_emb is not None
    assert _cloud_emb is not None

    q = _embedder.encode(text, convert_to_numpy=True).astype(np.float32, copy=False)
    q = q / (np.linalg.norm(q) + 1e-12)

    local_scores = _local_emb @ q
    cloud_scores = _cloud_emb @ q

    best_local_idx = int(np.argmax(local_scores))
    best_cloud_idx = int(np.argmax(cloud_scores))
    best_local = float(local_scores[best_local_idx])
    best_cloud = float(cloud_scores[best_cloud_idx])
    delta = best_cloud - best_local

    if best_cloud > best_local or delta > -MARGIN:
        mode = "CLOUD"
        confidence = best_cloud
        matched = _cloud_texts[best_cloud_idx]
    else:
        mode = "LOCAL"
        confidence = best_local
        matched = _local_texts[best_local_idx]

    return RouteResult(
        mode=mode,
        confidence=confidence,
        matched_anchor=matched,
        best_local=best_local,
        best_cloud=best_cloud,
        delta=delta,
    )

