"""Emotion classification using ONNX BERT model."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class EmotionResult:
    """Emotion classification result."""
    label: str
    raw_label: str
    score: float
    latency_s: float


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax over array."""
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


class EmotionClassifierONNX:
    """
    Emotion classifier (ONNX Runtime).
    
    - Uses onnxruntime + transformers tokenizer (no torch needed for classifier)
    - Reads labels from config.json(id2label) if available
    
    Args:
        model_dir: Directory that contains tokenizer files + ONNX file
        onnx_filename: Relative path to ONNX file (default: "onnx/model_quantized.onnx")
    """

    def __init__(self, model_dir: str, onnx_filename: str = "onnx/model_quantized.onnx"):
        self.model_dir = model_dir
        self.onnx_filename = onnx_filename
        self.available = False
        self.labels: List[str] = ["sadness", "joy", "love", "anger", "fear", "surprise"]

        # Lazy imports: Pi에서도 설치 부담/스타트업 시간을 줄입니다.
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except Exception as e:
            print(f"   [Emotion] Disabled (missing onnxruntime/transformers): {e}")
            return

        # 파일/디렉토리 존재 확인
        if not os.path.isdir(model_dir):
            print(f"   [Emotion] Disabled (missing model dir): {model_dir}")
            return

        # tokenizer 로드 (로컬 디렉토리 기준)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        except Exception as e:
            print(f"   [Emotion] Tokenizer load failed: {e}")
            return

        # labels 로드 (가능하면 config.json의 id2label 사용)
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                id2label = cfg.get("id2label")
                if isinstance(id2label, dict) and len(id2label) > 0:
                    # keys가 "0","1"... 형태(문자열 인덱스)일 수 있습니다.
                    self.labels = [id2label[str(i)] for i in range(len(id2label))]
            except Exception:
                pass

        # ONNX session 로드
        onnx_path = os.path.join(model_dir, onnx_filename)
        if not os.path.exists(onnx_path):
            # fallback: model_dir 내부에서 .onnx 파일을 탐색합니다.
            onnx_path = None
            for root, _, files in os.walk(model_dir):
                for fn in files:
                    if fn.endswith(".onnx"):
                        onnx_path = os.path.join(root, fn)
                        break
                if onnx_path:
                    break

        if onnx_path is None:
            print(f"   [Emotion] Disabled (missing .onnx): {model_dir}")
            return

        try:
            # Pi에서 순간 피크/발열을 줄이기 위해 스레드 수를 보수적으로 설정합니다.
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "4"))
            so.inter_op_num_threads = int(os.environ.get("ORT_INTER_OP_THREADS", "4"))
            self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"   [Emotion] ONNX session load failed: {e}")
            return

        # 입력 텐서 이름 캐시
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.available = True
        print(f"   [Emotion] Ready (onnx={os.path.basename(onnx_path)})")

    @staticmethod
    def _coarse_label(raw: str) -> str:
        """
        Map fine-grained GoEmotions labels to a smaller set for prompting.
        
        Args:
            raw: fine-grained label (e.g., 'admiration', 'annoyance')
        Returns:
            coarse label (sadness/joy/love/anger/fear/surprise/neutral)
        """
        mapping = {
            "sadness": "sadness",
            "disappointment": "sadness",
            "grief": "sadness",
            "remorse": "sadness",
            "joy": "joy",
            "amusement": "joy",
            "excitement": "joy",
            "optimism": "joy",
            "pride": "joy",
            "relief": "joy",
            "admiration": "joy",
            "approval": "joy",
            "gratitude": "joy",
            "love": "love",
            "caring": "love",
            "desire": "love",
            "anger": "anger",
            "annoyance": "anger",
            "disapproval": "anger",
            "disgust": "anger",
            "fear": "fear",
            "nervousness": "fear",
            "surprise": "surprise",
            "realization": "surprise",
            "confusion": "surprise",
            "curiosity": "surprise",
            "neutral": "neutral",
        }
        return mapping.get(raw, raw)

    def predict(self, text: str) -> Optional[EmotionResult]:
        """Predict emotion from text."""
        if not self.available:
            return None

        t0 = time.perf_counter()
        
        # tokenizer
        toks = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np",
        )

        feeds = {}
        for name in self.input_names:
            if name in toks:
                feeds[name] = toks[name]

        # run
        outputs = self.session.run(None, feeds)
        # logits: (num_labels,)
        logits = outputs[0][0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        raw_label = self.labels[idx] if idx < len(self.labels) else f"L{idx}"
        label = self._coarse_label(raw_label)
        score = float(probs[idx])

        return EmotionResult(label=label, raw_label=raw_label, score=score, latency_s=time.perf_counter() - t0)
