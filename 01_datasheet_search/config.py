"""Central configuration for the Phase 1 RAG pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "CH334_CH335_V2.5.pdf"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
# 표의 낱줄이 각각 청크가 되는 것을 막는 하한. chunk_size 를 넘지 않는다.
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "ch334u_datasheet")

# 근거 충분성 기준 (src/evidence.py). 값은 calibrate.py 로 평가셋에서 정했다 (2026.10.26).
#   용어 비율  정답 질문 최저 1.00, 문서에 없는 질문 최고 0.33 -> 가운데 0.67
#   1위 점수   정답 질문 최저 4.9 -> 절반 2.5 (점수만으로는 AEC-Q100 질문 12.5 를 못 막아 보조 기준)
EVIDENCE_MIN_SCORE = float(os.getenv("EVIDENCE_MIN_SCORE", "2.5"))
EVIDENCE_MIN_COVERAGE = float(os.getenv("EVIDENCE_MIN_COVERAGE", "0.67"))

# 답변 모델. "제공자:모델ID" 형식. 비교는 쉼표로 여러 개를 적는다.
# 기본값은 이 PC 에서 도는 로컬 모델이다 (Ollama, 비용 없음). 클라우드 모델은 키를 넣고 추가한다.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "ollama:gemma3:4b")
COMPARE_MODELS = [
    spec.strip()
    for spec in os.getenv(
        "COMPARE_MODELS", "ollama:gemma3:4b,ollama:nemotron-3-nano:4b@nothink"
    ).split(",")
    if spec.strip()
]
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "600"))

RESULTS_DIR = BASE_DIR / "results"
EVALSET_PATH = BASE_DIR / "tests" / "evaluation_questions.json"

DOCUMENT_METADATA = {
    "source": PDF_PATH.name,
    "manufacturer": "WCH",
    "part_number": "CH334U",
    "document_type": "datasheet",
    "revision": "V2.5",
}
