"""
DailyDialog 데이터셋을 사용하여 라우터 앵커를 생성하는 스크립트.

이 스크립트는 JSON 파일에서 후보를 읽어 앵커를 생성하고 임베딩을 생성합니다.
데이터 추출은 extract_dailydialog.py에서 수행합니다.

@author: Jaehyun, Park
@version: 2.1
"""

import json
import random
import os
import numpy as np
from typing import Optional

# NOTE:
# - sentence-transformers / numpy는 import가 무거울 수 있어
#   "임베딩을 실제로 생성할 때만" 늦게(import lazy) 불러옵니다.
_EMBEDDING_AVAILABLE: Optional[bool] = None

# Pi 라우팅용 권장 임베딩 모델 (영어 only)
# - all-MiniLM-L6-v2: 빠르고 가벼움(엣지 적합)
DEFAULT_EMBEDDING_MODEL = os.environ.get("ROUTER_EMBED_MODEL", "all-MiniLM-L6-v2")
# 저장 용량을 줄이기 위해 float16로 저장 (라우팅 시 float32로 계산)
SAVE_EMBEDDING_DTYPE = np.float16

# 상수 정의
NUM_LOCAL_ANCHORS = 80
NUM_CLOUD_FROM_DATASET = 30

# JSON 파일 경로 (dataset 디렉토리 내)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANCHORS_JSON = os.path.join(SCRIPT_DIR, "anchors.json")
DAILYDIALOG_CANDIDATES_JSON = os.path.join(SCRIPT_DIR, "dailydialog_candidates.json")
MANUAL_CLOUD_ANCHORS_JSON = os.path.join(SCRIPT_DIR, "manual_cloud_anchors.json")
ROUTER_ANCHORS_JSON = os.path.join(SCRIPT_DIR, "router_anchors.json")
LOCAL_EMB_PATH = os.path.join(SCRIPT_DIR, "local_anchors_embeddings.npy")
CLOUD_EMB_PATH = os.path.join(SCRIPT_DIR, "cloud_anchors_embeddings.npy")
METADATA_PATH = os.path.join(SCRIPT_DIR, "router_anchors_metadata.json")

def _bar():
    print("=" * 60)

def _step(title: str):
    print(f"\n## {title}")

def _info(msg: str):
    print(f"- {msg}")

def embedding_available() -> bool:
    """
    sentence-transformers 사용 가능 여부를 (한 번만) 체크합니다.
    """
    global _EMBEDDING_AVAILABLE
    if _EMBEDDING_AVAILABLE is not None:
        return _EMBEDDING_AVAILABLE
    try:
        import sentence_transformers  # noqa: F401
        _EMBEDDING_AVAILABLE = True
    except Exception:
        _EMBEDDING_AVAILABLE = False
    return _EMBEDDING_AVAILABLE

def load_prebuilt_anchors(json_path=ANCHORS_JSON):
    """
    anchors.json에서 (local_anchors, cloud_anchors)를 로드합니다.
    이 파일이 있으면 DailyDialog/수동 앵커 대신 우선 사용합니다.
    
    @param json_path: JSON 파일 경로
    @return: (local_anchors, cloud_anchors) 튜플 (없으면 (None, None))
    """
    if not os.path.exists(json_path):
        return None, None
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    local_anchors = data.get("local_anchors")
    cloud_anchors = data.get("cloud_anchors")
    
    if not isinstance(local_anchors, list) or not isinstance(cloud_anchors, list):
        raise ValueError(f"{json_path} must contain 'local_anchors' and 'cloud_anchors' as lists.")
    
    # 문자열만 남기고 정리
    local_anchors = [str(x).strip() for x in local_anchors if str(x).strip()]
    cloud_anchors = [str(x).strip() for x in cloud_anchors if str(x).strip()]
    
    return local_anchors, cloud_anchors

def load_dailydialog_candidates(json_path=DAILYDIALOG_CANDIDATES_JSON):
    """
    DailyDialog에서 추출한 후보를 JSON 파일에서 로드합니다.
    
    @param json_path: JSON 파일 경로
    @return: (local_candidates, cloud_candidates) 튜플
    """
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} 파일을 찾을 수 없습니다.")
        print("먼저 extract_dailydialog.py를 실행하여 후보를 추출하세요.")
        return [], []
    
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    local_candidates = data.get("local_candidates", [])
    cloud_candidates = data.get("cloud_candidates", [])
    
    return local_candidates, cloud_candidates

def load_manual_cloud_anchors(json_path=MANUAL_CLOUD_ANCHORS_JSON):
    """
    JSON 파일에서 수동 정의된 Cloud 앵커를 로드합니다.
    
    @param json_path: JSON 파일 경로
    @return: Cloud 앵커 리스트
    """
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} 파일을 찾을 수 없습니다. 빈 리스트를 반환합니다.")
        return []
    
    with open(json_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    # 모든 카테고리의 앵커를 하나의 리스트로 합침
    manual_anchors = []
    for category, anchors in data.items():
        manual_anchors.extend(anchors)
    
    return manual_anchors

# 메인 로직
_bar()
print("라우터 앵커 생성 시작")
_info(f"작업 디렉토리: {SCRIPT_DIR}")
_bar()

# 0. anchors.json 우선 로드
_step("0) anchors.json 확인")
_info(f"입력 후보(우선): {ANCHORS_JSON}")
prebuilt_local, prebuilt_cloud = load_prebuilt_anchors()

if prebuilt_local is not None and prebuilt_cloud is not None:
    _info(f"anchors.json 사용: Local {len(prebuilt_local)}개 / Cloud {len(prebuilt_cloud)}개")
    local_anchors = prebuilt_local
    cloud_anchors = prebuilt_cloud
else:
    _info("anchors.json 없음 → 기존 파이프라인 사용")

    # 1. DailyDialog 후보 로드
    _step("1) 후보 로드 (dailydialog_candidates.json)")
    _info(f"입력 후보: {DAILYDIALOG_CANDIDATES_JSON}")
    local_candidates, cloud_candidates = load_dailydialog_candidates()
    _info(f"Local 후보: {len(local_candidates)}개")
    _info(f"Cloud 후보: {len(cloud_candidates)}개")

    # 2. Local 앵커 샘플링
    _step("2) Local 앵커 샘플링")
    num_local = min(NUM_LOCAL_ANCHORS, len(local_candidates))
    local_anchors = random.sample(local_candidates, num_local) if num_local > 0 else []
    _info(f"샘플링: {num_local}개 (max={NUM_LOCAL_ANCHORS})")

    # 3. Cloud 앵커: DailyDialog에서 추출 + 수동 정의
    cloud_anchors = []

    # DailyDialog에서 추출한 지식 검색 케이스 추가
    if len(cloud_candidates) > 0:
        _step("3) Cloud 앵커 샘플링 (from candidates)")
        num_from_dataset = min(NUM_CLOUD_FROM_DATASET, len(cloud_candidates))
        cloud_anchors.extend(random.sample(cloud_candidates, num_from_dataset))
        _info(f"샘플링: {num_from_dataset}개 (max={NUM_CLOUD_FROM_DATASET})")
    else:
        _step("3) Cloud 앵커 샘플링 (from candidates)")
        _info("Cloud 후보가 0개라 샘플링을 건너뜁니다.")

    # 수동 정의 Cloud 앵커 추가
    _step("4) 수동 Cloud 앵커 로드 (manual_cloud_anchors.json)")
    _info(f"입력 후보: {MANUAL_CLOUD_ANCHORS_JSON}")
    manual_cloud_anchors = load_manual_cloud_anchors()
    _info(f"로드: {len(manual_cloud_anchors)}개")
    cloud_anchors.extend(manual_cloud_anchors)

def generate_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    텍스트 리스트에 대한 임베딩을 생성합니다.
    
    @param texts: 임베딩을 생성할 텍스트 리스트
    @param model_name: 사용할 임베딩 모델 이름
    @return: 임베딩 배열 (numpy array)
    """
    if not embedding_available():
        return None

    # Lazy imports (startup 속도 개선)
    from sentence_transformers import SentenceTransformer
    
    _step("6) 임베딩 생성")
    _info(f"임베딩 모델 로드: {model_name}")
    _info("처음 실행이면 모델 다운로드로 시간이 걸릴 수 있습니다.")
    model = SentenceTransformer(model_name)
    
    _info(f"임베딩 생성 중... (N={len(texts)})")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings

# 5. JSON 저장
_step("5) router_anchors.json 저장")
_info(f"출력: {ROUTER_ANCHORS_JSON}")
data = {
    "local_anchors": local_anchors,
    "cloud_anchors": cloud_anchors
}

with open(ROUTER_ANCHORS_JSON, "w", encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
_info("저장 완료")

# 6. 임베딩 생성 (선택사항)
if embedding_available():
    # sentence-transformers는 임베딩 생성 시에만 import
    from sentence_transformers import SentenceTransformer

    _step("6) 임베딩 생성")
    _info(f"임베딩 모델: {DEFAULT_EMBEDDING_MODEL}")
    _info("처음 실행이면 모델 다운로드로 시간이 걸릴 수 있습니다.")
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-12)

    # Local 앵커 임베딩
    if len(local_anchors) > 0:
        _info(f"Local 임베딩 생성... (N={len(local_anchors)})")
        local_embeddings = model.encode(local_anchors, show_progress_bar=True, convert_to_numpy=True)
        local_embeddings = _normalize(local_embeddings).astype(SAVE_EMBEDDING_DTYPE, copy=False)
        np.save(LOCAL_EMB_PATH, local_embeddings)
        _info(f"Local 임베딩 저장: {LOCAL_EMB_PATH} / shape={local_embeddings.shape} / dtype={local_embeddings.dtype}")

    # Cloud 앵커 임베딩
    if len(cloud_anchors) > 0:
        _info(f"Cloud 임베딩 생성... (N={len(cloud_anchors)})")
        cloud_embeddings = model.encode(cloud_anchors, show_progress_bar=True, convert_to_numpy=True)
        cloud_embeddings = _normalize(cloud_embeddings).astype(SAVE_EMBEDDING_DTYPE, copy=False)
        np.save(CLOUD_EMB_PATH, cloud_embeddings)
        _info(f"Cloud 임베딩 저장: {CLOUD_EMB_PATH} / shape={cloud_embeddings.shape} / dtype={cloud_embeddings.dtype}")
    
    # 메타데이터 저장
    metadata = {
        "local_count": len(local_anchors),
        "cloud_count": len(cloud_anchors),
        "local_embedding_shape": list(local_embeddings.shape) if len(local_anchors) > 0 else None,
        "cloud_embedding_shape": list(cloud_embeddings.shape) if len(cloud_anchors) > 0 else None,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_normalized": True,
        "embedding_dtype": str(SAVE_EMBEDDING_DTYPE)
    }
    
    with open(METADATA_PATH, "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    _info(f"메타데이터 저장: {METADATA_PATH}")
else:
    _step("6) 임베딩 생성")
    _info("건너뜀: sentence-transformers 미설치")
    _info("설치 후 재실행하면 .npy 임베딩이 생성됩니다: pip install sentence-transformers")

_step("완료 요약")
_info(f"Local 앵커: {len(local_anchors)}개")
_info(f"Cloud 앵커: {len(cloud_anchors)}개")
_info(f"생성 파일: {ROUTER_ANCHORS_JSON}")
if embedding_available():
    _info(f"생성 파일: {LOCAL_EMB_PATH}")
    _info(f"생성 파일: {CLOUD_EMB_PATH}")
    _info(f"생성 파일: {METADATA_PATH}")
_bar()
print("라즈베리파이에 필요한 파일만 복사하면 됩니다.")
