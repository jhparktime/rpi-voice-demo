"""Route short-vs-long response mode for Gemini prompts."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


SHORT_ANCHORS = [
    "quick answer, short confirmation, yes/no, direct reply",
    "user asks for a short response",
    "one-line answer requested",
    "brief clarification request",
    "simple fact with no deep steps",
]

LONG_ANCHORS = [
    "explain in detail, teach, elaborate, step-by-step",
    "compare options with trade-offs",
    "reason why, pros and cons, analysis",
    "compare two things or multiple versions",
    "show example plan or checklist",
    "complex technical or procedural breakdown",
]

LONG_HINT_KEYWORDS = [
    "explain",
    "details",
    "why",
    "compare",
    "pros and cons",
    "step by step",
    "step-by-step",
    "analyze",
    "how to",
    "how does",
    "longer",
    "deep",
]

SHORT_HINT_KEYWORDS = [
    "quick",
    "just the answer",
    "just answer",
    "in short",
    "short answer",
    "one sentence",
    "briefly",
]


_ANCHOR_MODEL = os.environ.get("ROUTER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_MODEL: Optional[SentenceTransformer] = None
_SHORT_VEC: Optional[np.ndarray] = None
_LONG_VEC: Optional[np.ndarray] = None


@dataclass
class RouteDecision:
    mode: str
    confidence: float
    reason: str
    short_score: float
    long_score: float
    top1_score: float
    top2_score: float

    def as_dict(self) -> dict[str, str]:
        return {"mode": self.mode, "reason": self.reason}


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _prepare_models() -> None:
    global _MODEL, _SHORT_VEC, _LONG_VEC
    if _MODEL is not None and _SHORT_VEC is not None and _LONG_VEC is not None:
        return

    _MODEL = SentenceTransformer(_ANCHOR_MODEL)
    short_emb = _MODEL.encode(SHORT_ANCHORS, convert_to_numpy=True).astype(np.float32)
    long_emb = _MODEL.encode(LONG_ANCHORS, convert_to_numpy=True).astype(np.float32)
    _SHORT_VEC = _norm(short_emb)
    _LONG_VEC = _norm(long_emb)


def _keyword_mode(text: str) -> Optional[str]:
    norm = (text or "").lower()
    for token in LONG_HINT_KEYWORDS:
        if token in norm:
            return "LONG"
    for token in SHORT_HINT_KEYWORDS:
        if token in norm:
            return "SHORT"
    return None


def route_query(
    text: str,
    min_score: Optional[float] = None,
    margin: Optional[float] = None,
) -> RouteDecision:
    """
    Route one input query to SHORT or LONG mode.

    The decision is deterministic and returns a reason useful for logging.
    """
    safe_text = (text or "").strip()
    keyword_mode = _keyword_mode(safe_text)
    if not safe_text:
        return RouteDecision(
            mode="SHORT",
            confidence=1.0,
            reason="empty_input",
            short_score=0.0,
            long_score=0.0,
            top1_score=0.0,
            top2_score=0.0,
        )

    min_score = float(os.environ.get("ROUTER_MIN_SCORE", "0.17") if min_score is None else min_score)
    margin = float(os.environ.get("ROUTER_MARGIN", "0.03") if margin is None else margin)

    if keyword_mode is not None:
        confidence = 0.95 if keyword_mode == "LONG" else 0.90
        return RouteDecision(
            mode=keyword_mode,
            confidence=confidence,
            reason=f"keyword_hint:{keyword_mode.lower()}",
            short_score=1.0 if keyword_mode == "SHORT" else 0.0,
            long_score=1.0 if keyword_mode == "LONG" else 0.0,
            top1_score=confidence,
            top2_score=1.0 - confidence,
        )

    try:
        _prepare_models()
        assert _MODEL is not None and _SHORT_VEC is not None and _LONG_VEC is not None
        q = _MODEL.encode(safe_text, convert_to_numpy=True).astype(np.float32)
        q = _norm(q)
        short_scores = _SHORT_VEC @ q
        long_scores = _LONG_VEC @ q
        short_top1 = float(np.max(short_scores))
        long_top1 = float(np.max(long_scores))
        all_scores: List[Tuple[float, str]] = [(short_top1, "SHORT"), (long_top1, "LONG")]
        all_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_mode = all_scores[0]
        second_score = all_scores[1][0] if len(all_scores) > 1 else 0.0
        if best_score < min_score or (best_score - second_score) < margin:
            reason = f"uncertain_best_score={best_score:.3f}_delta={best_score - second_score:.3f}"
            return RouteDecision(
                mode="SHORT",
                confidence=max(0.0, best_score),
                reason=reason,
                short_score=short_top1,
                long_score=long_top1,
                top1_score=best_score,
                top2_score=second_score,
            )

        reason = f"anchor_sim_{best_mode.lower()}_score={best_score:.3f}_delta={best_score - second_score:.3f}"
        return RouteDecision(
            mode=best_mode,
            confidence=best_score,
            reason=reason,
            short_score=short_top1,
            long_score=long_top1,
            top1_score=best_score,
            top2_score=second_score,
        )
    except Exception as exc:  # noqa: BLE001
        return RouteDecision(
            mode="SHORT",
            confidence=0.0,
            reason=f"router_error:{exc!r}",
            short_score=0.0,
            long_score=0.0,
            top1_score=0.0,
            top2_score=0.0,
        )
