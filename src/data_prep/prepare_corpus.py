import json
import lzma
from pathlib import Path
from typing import Iterator, Dict

from src.config import (
    CUAD_FULL_TEXT_DIR,
    CASELAW_JSONL_XZ,
    CORPUS_JSONL,
)


def iter_cuad_docs() -> Iterator[Dict]:
    """
    Iterate over CUAD contracts as documents.
    Assumes one .txt file per contract in CUAD_FULL_TEXT_DIR.
    """
    txt_dir = CUAD_FULL_TEXT_DIR
    if not txt_dir.exists():
        raise FileNotFoundError(f"CUAD full_contract_txt directory not found: {txt_dir}")

    for path in sorted(txt_dir.glob("*.txt")):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        doc_id = f"cuad::{path.stem}"
        title = path.stem.replace("_", " ")

        yield {
            "doc_id": doc_id,
            "source": "cuad",
            "title": title,
            "text": text,
        }


def extract_caselaw_text(record: Dict) -> str:
    """
    Extract main opinion text from a single caselaw JSON record.

    NOTE: The exact structure depends on the Kaggle dataset.
    This is a *guess* based on typical CaseLaw / CourtListener format.
    If it errors, print(record.keys()) and adjust accordingly.
    """
    # Try a few common patterns:
    casebody = record.get("casebody") or record.get("case_body") or {}

    # Example CaseLaw Access Project structure:
    data = casebody.get("data") or {}
    opinions = data.get("opinions") or []

    if opinions and isinstance(opinions, list):
        # Concatenate all opinions' text
        return "\n\n".join(o.get("text", "") for o in opinions if isinstance(o, dict))

    # Fallback: maybe the text is directly stored in "casebody"
    if isinstance(casebody, dict) and "text" in casebody:
        return casebody["text"]

    # Last resort: try 'text' on the top level
    return record.get("text", "")


def iter_caselaw_docs() -> Iterator[Dict]:
    """
    Iterate over caselaw cases as documents from compressed JSONL (.xz).
    """
    path = CASELAW_JSONL_XZ
    if not path.exists():
        raise FileNotFoundError(f"Caselaw JSONL.xz file not found: {path}")

    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at line {line_idx}")
                continue

            # Try to build a stable doc_id and title
            case_id = record.get("id") or record.get("case_id") or f"case_{line_idx}"
            name = record.get("name") or record.get("case_name") or f"Case {line_idx}"

            text = extract_caselaw_text(record)
            if not text.strip():
                # Skip empty texts
                continue

            yield {
                "doc_id": f"caselaw::{case_id}",
                "source": "caselaw",
                "title": name,
                "text": text,
            }


def build_corpus(out_path: Path = CORPUS_JSONL) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        # CUAD docs
        for doc in iter_cuad_docs():
            out_f.write(json.dumps(doc) + "\n")
            n += 1

        # Caselaw docs
        for doc in iter_caselaw_docs():
            out_f.write(json.dumps(doc) + "\n")
            n += 1

    print(f"Wrote {n} documents to {out_path}")


if __name__ == "__main__":
    build_corpus()
