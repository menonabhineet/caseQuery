from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
CUAD_DIR = DATA_DIR / "cuad"
CASELAW_DIR = DATA_DIR / "caselaw"

CUAD_FULL_TEXT_DIR = CUAD_DIR / "full_contract_txt"
CASELAW_JSONL_XZ = CASELAW_DIR / "text.data.jsonl.xz"

CORPUS_JSONL = DATA_DIR / "corpus.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl" 

EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"
CHUNKS_META_JSONL = DATA_DIR / "chunks_meta.jsonl"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
