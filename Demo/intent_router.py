"""Intent routing and bridge templates for CLOUD/LOCAL routing."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# Bridge templates for CLOUD routing
BRIDGE_TEMPLATES_DEFAULT = [
    "Got it—give me a moment to check that for you.",
    "Okay, I hear you—let me look into that and get back to you.",
    "Alright—let me double-check that so I don't mislead you.",
    "Thanks, I'm on it—give me a second to verify the details.",
]

"""
Broad categories for CLOUD bridge templates (avoid overly specific types like "sql").
"""
BRIDGE_TEMPLATES_BY_TYPE: Dict[str, List[str]] = {
    "explain": [
        "Got it—I'll put together a clear, simple explanation and get back to you in a moment.",
        "Okay—I'll think through a clean explanation and then reply.",
    ],
    "summarize": [
        "Got it—I'll read that and summarize the key points in a moment.",
        "Okay—I'll skim this carefully and then summarize it for you.",
    ],
    "translate": [
        "Got it—I'll double-check the translation and get back to you in a moment.",
        "Okay—I'll verify the translation carefully and then reply.",
    ],
    "compute": [
        "Got it—I'll work through the numbers carefully and get back to you in a moment.",
        "Okay—I'll calculate that and then reply.",
    ],
    "create": [
        "Got it—I'll draft a solid version of that and get back to you in a moment.",
        "Okay—I'll put together a clean draft and then reply.",
    ],
    "code": [
        "Got it—I'll think through the implementation and get back to you in a moment.",
        "Okay—I'll draft a workable solution and then reply.",
    ],
}

"""
Intent anchors for bridge-type selection.
NOTE:
- These are intent descriptions, not keyword rules.
- We choose the best type by embedding similarity.
"""
BRIDGE_INTENT_ANCHORS: Dict[str, List[str]] = {
    "explain": [
        "Explain a concept in simple terms.",
        "Give a clear explanation of how something works.",
        "Answer a conceptual question with a clear explanation.",
    ],
    "summarize": [
        "Summarize a piece of text into key points.",
        "Create a short summary of a paragraph.",
        "Condense a long text into a brief summary.",
    ],
    "translate": [
        "Translate a sentence into another language.",
        "Provide a translation for a given sentence.",
        "Rewrite a sentence in a different language.",
    ],
    "compute": [
        "Solve a math problem or compute a numeric result.",
        "Calculate a value from given numbers.",
        "Convert units or compute a numeric answer.",
    ],
    "create": [
        "Write a draft such as an email, message, or short document.",
        "Create a plan, checklist, or itinerary.",
        "Generate a structured response like steps or a template.",
    ],
    "code": [
        "Write code to implement a function or script.",
        "Generate a programming solution (including SQL) for a task.",
        "Draft a technical implementation or query.",
    ],
    "default": [
        "Handle a complex request that needs careful checking.",
        "Answer a knowledge-heavy question after verifying details.",
    ],
}

# --- Simple LOCAL vs CLOUD intent anchors (for Demo) ---

EASY_LOCAL_ANCHORS: List[str] = [
    "Casual small talk or chitchat.",
    "Short reaction to how I'm feeling.",
    "Simple check-in about my day.",
    "Light emotional support for everyday stress.",
]

COMPLEX_CLOUD_ANCHORS: List[str] = [
    "Technical question about programming or math.",
    "Question that needs accurate factual knowledge.",
    "Complex explanation about how something works in detail.",
    "Help with code, SQL, or debugging a problem.",
]

_intent_embedder: Optional[SentenceTransformer] = None
_easy_vec: Optional[np.ndarray] = None
_complex_vec: Optional[np.ndarray] = None


def normalize_vec(v: np.ndarray) -> np.ndarray:
    """
    L2-normalize a single vector.
    
    Args:
        v: 1D vector
    Returns:
        normalized vector
    """
    n = np.linalg.norm(v) + 1e-12
    return v / n


class IntentRouter:
    """
    Intent router for selecting bridge templates based on embedding similarity.
    
    Uses sentence embeddings to match user intent to bridge template types.
    """
    
    def __init__(self, embedder: SentenceTransformer, precompute: bool = False):
        """
        Initialize intent router.
        
        Args:
            embedder: SentenceTransformer instance for encoding
            precompute: If True, precompute intent embeddings at init (faster first call, higher init load)
        """
        self.embedder = embedder
        self.bridge_types = list(BRIDGE_INTENT_ANCHORS.keys())
        self.bridge_type_to_templates = dict(BRIDGE_TEMPLATES_BY_TYPE)
        self.bridge_type_to_templates["default"] = BRIDGE_TEMPLATES_DEFAULT
        
        # Flatten intent anchors
        flat_texts: List[str] = []
        flat_types: List[str] = []
        for t, anchors in BRIDGE_INTENT_ANCHORS.items():
            for a in anchors:
                flat_texts.append(a)
                flat_types.append(t)
        
        self.bridge_intent_texts = flat_texts
        self.bridge_intent_types = flat_types
        self.bridge_intent_emb: Optional[np.ndarray] = None
        
        if precompute:
            self._compute_intent_embeddings()
    
    def _compute_intent_embeddings(self) -> None:
        """Compute and normalize intent embeddings."""
        embs = self.embedder.encode(self.bridge_intent_texts, convert_to_numpy=True)
        embs = embs.astype(np.float32, copy=False)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        self.bridge_intent_emb = embs
    
    def select_bridge_template(self, user_text: str) -> str:
        """
        Select bridge template based on user intent.
        
        Args:
            user_text: User input text
        Returns:
            Selected bridge template string
        """
        # Lazy init if not precomputed
        if self.bridge_intent_emb is None:
            self._compute_intent_embeddings()
        
        # Encode user text
        q = self.embedder.encode(user_text, convert_to_numpy=True).astype(np.float32, copy=False)
        q = normalize_vec(q)
        
        # Compute similarity scores
        scores = self.bridge_intent_emb @ q
        best_idx = int(np.argmax(scores))
        best_type = self.bridge_intent_types[best_idx] if best_idx < len(self.bridge_intent_types) else "default"
        
        # Select template from pool
        pool = self.bridge_type_to_templates.get(best_type, BRIDGE_TEMPLATES_DEFAULT)
        # Use hash for deterministic selection
        out = pool[abs(hash(f"{best_type}:{user_text}")) % len(pool)]
        return out


def _ensure_intent_embedder() -> None:
    """Lazy-init SentenceTransformer and simple LOCAL/CLOUD prototype vectors."""
    global _intent_embedder, _easy_vec, _complex_vec
    if _intent_embedder is not None and _easy_vec is not None and _complex_vec is not None:
        return

    _intent_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    easy_embs = _intent_embedder.encode(EASY_LOCAL_ANCHORS, convert_to_numpy=True).astype(np.float32, copy=False)
    complex_embs = _intent_embedder.encode(COMPLEX_CLOUD_ANCHORS, convert_to_numpy=True).astype(np.float32, copy=False)
    easy_embs = easy_embs / (np.linalg.norm(easy_embs, axis=1, keepdims=True) + 1e-12)
    complex_embs = complex_embs / (np.linalg.norm(complex_embs, axis=1, keepdims=True) + 1e-12)
    _easy_vec = easy_embs.mean(axis=0)
    _complex_vec = complex_embs.mean(axis=0)


def classify_intent_easy_or_complex(text: str, margin: float = 0.03) -> str:
    """
    Classify intent into LOCAL (easy) or CLOUD (complex) using simple embeddings.

    Returns:
        "LOCAL" for easy/local-style requests, "CLOUD" for complex/knowledge-heavy ones.
    """
    if not text or not text.strip():
        return "LOCAL"

    _ensure_intent_embedder()
    assert _intent_embedder is not None
    assert _easy_vec is not None
    assert _complex_vec is not None

    q = _intent_embedder.encode(text, convert_to_numpy=True).astype(np.float32, copy=False)
    q = normalize_vec(q)
    easy_score = float(_easy_vec @ q)
    complex_score = float(_complex_vec @ q)

    # If complex clearly wins (or within small margin), treat as CLOUD
    if complex_score > easy_score and (complex_score - easy_score) > margin:
        return "CLOUD"
    return "LOCAL"
