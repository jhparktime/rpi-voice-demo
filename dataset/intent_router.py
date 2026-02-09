import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RouteResult:
    decision: str  # "LOCAL" | "CLOUD"
    confidence: float
    matched_anchor: str
    latency_s: float
    best_local: float
    best_cloud: float
    delta: float


class IntentRouter:
    """
    Raspberry Pi용 의도 라우터 (라우팅 런타임)
    - M1에서 생성한 router_anchors.json + *.npy 임베딩을 로드해서 사용
    - anchors 임베딩은 미리 정규화되어 있다고 가정 (make_anchor.py 기준)
    - Pi에서는 입력 1문장만 임베딩 → dot-product로 cosine similarity
    """

    def __init__(
        self,
        router_anchors_json: str = "router_anchors.json",
        local_emb_path: str = "local_anchors_embeddings.npy",
        cloud_emb_path: str = "cloud_anchors_embeddings.npy",
        metadata_path: str = "router_anchors_metadata.json",
        model_name: Optional[str] = None,
        margin: float = 0.03,
    ):
        self.margin = margin

        t0 = time.time()

        # 1) 텍스트 로드 (디버깅용: matched anchor 출력)
        with open(router_anchors_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.local_texts: List[str] = data.get("local_anchors", [])
        self.cloud_texts: List[str] = data.get("cloud_anchors", [])
        if not self.local_texts or not self.cloud_texts:
            raise ValueError("router_anchors.json must contain non-empty local_anchors and cloud_anchors.")

        # 2) 임베딩 로드
        self.local_emb = np.load(local_emb_path)
        self.cloud_emb = np.load(cloud_emb_path)

        if self.local_emb.shape[0] != len(self.local_texts):
            raise ValueError("local embeddings count does not match local_anchors count.")
        if self.cloud_emb.shape[0] != len(self.cloud_texts):
            raise ValueError("cloud embeddings count does not match cloud_anchors count.")

        # 3) 메타데이터 로드 (모델명 확인)
        embed_model = None
        normalized = False
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            embed_model = meta.get("embedding_model")
            normalized = bool(meta.get("embedding_normalized", False))

        self.embedding_model_name = model_name or embed_model or "all-MiniLM-L6-v2"
        self.embeddings_normalized = normalized

        # 4) Pi에서 query 임베딩용 모델 로드
        # NOTE: Pi에서 torch 설치가 부담이면 ONNX 버전으로 교체 가능(추후)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.embedding_model_name, device="cpu")

        self.init_latency_s = time.time() - t0

    @staticmethod
    def _normalize_vec(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def route(self, user_text: str) -> RouteResult:
        t0 = time.time()

        q = self.model.encode(user_text, convert_to_numpy=True)
        q = q.astype(np.float32, copy=False)
        q = self._normalize_vec(q)

        local = self.local_emb.astype(np.float32, copy=False)
        cloud = self.cloud_emb.astype(np.float32, copy=False)

        if not self.embeddings_normalized:
            # 안전장치: 혹시 정규화 안 되어 있으면 여기서 정규화
            local = local / (np.linalg.norm(local, axis=1, keepdims=True) + 1e-12)
            cloud = cloud / (np.linalg.norm(cloud, axis=1, keepdims=True) + 1e-12)

        # cosine similarity (normalized면 dot = cosine)
        local_scores = local @ q
        cloud_scores = cloud @ q

        best_local_idx = int(np.argmax(local_scores))
        best_cloud_idx = int(np.argmax(cloud_scores))
        best_local = float(local_scores[best_local_idx])
        best_cloud = float(cloud_scores[best_cloud_idx])
        delta = best_cloud - best_local

        # decision: cloud가 조금이라도 우세하거나(>0),
        # 마진이 작을 때는 보수적으로 cloud로 보냄(엣지 오답 방지)
        if best_cloud > best_local or delta > -self.margin:
            decision = "CLOUD"
            confidence = best_cloud
            matched = self.cloud_texts[best_cloud_idx]
        else:
            decision = "LOCAL"
            confidence = best_local
            matched = self.local_texts[best_local_idx]

        return RouteResult(
            decision=decision,
            confidence=confidence,
            matched_anchor=matched,
            latency_s=time.time() - t0,
            best_local=best_local,
            best_cloud=best_cloud,
            delta=delta,
        )


if __name__ == "__main__":
    print("IntentRouter (Pi runtime) starting...")
    router = IntentRouter(
        router_anchors_json="router_anchors.json",
        local_emb_path="local_anchors_embeddings.npy",
        cloud_emb_path="cloud_anchors_embeddings.npy",
        metadata_path="router_anchors_metadata.json",
        margin=0.03,
    )
    print(f"Ready. init_latency={router.init_latency_s:.2f}s, model={router.embedding_model_name}")
    print("Type 'q' to quit.")

    while True:
        text = input("User Input >> ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        r = router.route(text)
        print(f"[{r.decision}] conf={r.confidence:.4f} delta={r.delta:.4f} time={r.latency_s*1000:.1f}ms")
        print(f"match: {r.matched_anchor}")

