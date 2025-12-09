
# CaseQuery

A retrieval-augmented generation (RAG) system for legal question answering over case law and contracts. CaseQuery combines dense embeddings (FAISS) and traditional retrieval baselines (BM25) with large language models to provide grounded, verifiable answers to legal questions.

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Datasets](#datasets)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)


## Introduction

The growing availability of digital legal documents has created both an opportunity and a challenge for practitioners: while case law and contracts are more accessible than ever, locating the precise passages that answer a focused legal question remains slow and error-prone. 
CaseQuery addresses this gap by building a comprehensive RAG system for legal question answering. The system combines multiple retrieval strategies - BM25, intra-document retrieval, dense embeddings, and hybrid sparse+dense retrieval with reranking, to systematically evaluate how retrieval choices affect both retrieval quality and downstream answer reliability. Every answer is explicitly linked to underlying cases, clauses, or sections, enabling users to verify every claim. A rigorous evaluation pipeline  quantifies how retrieval choices shape the trustworthiness of legal LLM assistants.

## Problem Statement

Given a corpus of legal documents (contracts and case law) and a natural language question, design a system that:
- Retrieves relevant passages from the document corpus
- Generates a grounded answer that minimizes hallucinations
- Maximizes alignment with the underlying legal text
- Evaluates performance against non-retrieval baselines

## Datasets

CaseQuery uses two complementary legal datasets:

### 1. Contract Understanding Atticus Dataset (CUAD)
- **Source**: [CUAD Dataset](https://zenodo.org/records/4595826)
- **Description**: A corpus of 13,000+ labels across 510 commercial legal contracts manually labeled to identify 41 categories of important clauses that lawyers look for when reviewing contracts in connection with corporate transactions.

### 2. CaseLaw Access Project (CAP)
- **Source**: https://www.kaggle.com/datasets/harvardlil/caselaw-dataset-illinois/data
- **Description**: Official, book-published state and federal United States case law through 2020. This project uses a subset from the Illinois Jurisdiction.

## Architecture

The system consists of five integrated layers:

### Data Layer
- Unified corpus from CUAD and CaseLaw datasets
- Chunked passages with overlapping windows for better retrieval

### Retrieval Layer
- **Dense Retriever**: FAISS-based retrieval using sentence embeddings
- **BM25 Baseline**: Traditional term-frequency baseline for comparison

### Generation Layer
- LLM via OpenRouter (meta-llama)
- Grounded prompting with retrieved context

### Evaluation Layer
- Retrieval metrics: Recall@k, Mean Reciprocal Rank (MRR)
- Generation metrics: ROUGE-L, BERTScore

### Interface Layer
- Interactive CLI demo for both RAG and zero-shot modes

## Project Structure

```
caseQuery/
├── app/                          # Application interface
│   ├── __init__.py
│   └── cli.py                   # Interactive CLI demo
├── data/                         # Data artifacts
│   ├── eval_queries.jsonl       # Evaluation queries
│   └── generation_eval.jsonl    # Generation evaluation examples
├── notebooks/                    # Test and analysis scripts
│   ├── test_bm25.py
│   ├── test_rag.py
│   ├── test_retriever.py
│   └── test_zero_shot.py
├── src/                         # Core source code
│   ├── config.py               # Configuration settings
│   ├── baselines/              # Baseline implementations
│   │   ├── bm25_retriever.py   # BM25 baseline retriever
│   │   └── zero_shot_llm.py    # Zero-shot LLM baseline
│   ├── data_prep/              # Data preparation
│   │   ├── prepare_corpus.py   # Unify CUAD and CaseLaw data
│   │   └── chunk_corpus.py     # Chunk documents
│   ├── evaluation/             # Evaluation metrics
│   │   ├── generation_eval.py  # Generation quality metrics
│   │   └── retrieval_eval.py   # Retrieval metrics (Recall@k, MRR)
│   ├── generator/              # Generation components
│   │   ├── llm_client.py       # LLM API wrapper
│   │   ├── prompting.py        # Prompt construction
│   │   └── rag_pipeline.py     # RAG orchestration
│   └── retriever/              # Retrieval components
│       ├── embed_and_index.py  # Embedding and indexing
│       └── faiss_retriever.py  # FAISS retriever
└── README.md                    # This file
```

## Key Components

### `src/retriever/faiss_retriever.py`
Implements dense semantic retrieval using FAISS with all-MiniLM-L6-v2 embeddings.

### `src/baselines/bm25_retriever.py`
Traditional term-frequency baseline using BM25Okapi for comparison.

### `src/generator/rag_pipeline.py`
Orchestrates the complete RAG workflow: retrieval → prompting → generation.

### `src/baselines/zero_shot_llm.py`
Baseline LLM queries without retrieval or context.

### `src/evaluation/retrieval_eval.py` and `src/evaluation/generation_eval.py`
Evaluation scripts for systematic metric collection and comparison.

## Methodology

### Corpus Construction
The corpus construction process consists of two stages:

1. **Unified Corpus**: `src/data_prep/prepare_corpus.py` reads data from both CUAD and CaseLaw datasets and writes a unified `corpus.jsonl` file.
2. **Chunking**: `src/data_prep/chunk_corpus.py` splits each document into overlapping word-based chunks and writes `chunks.jsonl`.

### Dense Retriever (FAISS)
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Dimension**: 384-dimensional normalized embeddings
- **Process**:
  1. `src/retriever/embed_and_index.py` loads chunks from `data/chunks.jsonl`
  2. Computes embeddings in batches
  3. Saves `embeddings.npy`, `chunks_meta.jsonl`, and `faiss_index.bin`
- **Query**: Encodes input question using the same embedding model and performs FAISS search, returning top-k RetrievedChunk objects

### BM25 Retriever Baseline
- **Library**: rank-bm25 (BM25Okapi implementation)
- **Process**:
  1. Loads chunk records from `data/chunks.jsonl`
  2. Lowercases and tokenizes text into word tokens
  3. Builds in-memory BM25 index
- **Query**: Scores against all indexed chunks, returns top-k RetrievedChunk objects ranked by BM25 relevance scores

### LLM Client and Prompting
- **LLM**: meta-llama model served via OpenRouter API
- **System Prompt**: Instructs the model to:
  - Act as a careful legal assistant
  - Rely only on provided context
  - Avoid hallucinations
  - Always cite passages used
- **User Prompt**: Includes question + retrieved passages with metadata (chunk ID, source, title)

### RAG Pipeline
The `LegalRAG` class orchestrates the full pipeline:
1. Calls FaissRetriever to fetch top-k relevant chunks
2. Builds a user prompt embedding passages under numbered tags
3. Sends prompt with legal system prompt to meta-llama
4. Returns RAGAnswer with both natural-language response and supporting evidence

### Zero-Shot LLM Baseline
The `ZeroShotLegalQA` class provides a baseline without retrieval:
- Sends only the question to the LLM
- Uses a generic system prompt
- Returns plain natural-language answer without citations
- Enables direct comparison between grounded RAG and unguided LLM behavior

### CLI Demo
Interactive command-line interface (`app/cli.py`) supporting:
- **RAG Mode**: Instantiates LegalRAG, displays answers with source passages
- **Zero-Shot Mode**: Uses ZeroShotLegalQA for side-by-side comparison

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements below)
- OpenRouter API key for LLM access

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd caseQuery
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenRouter API key (add to environment or config)

### Data Preparation

```bash
download both the datasets mentioned above, unzip and  paste both folders within "data" folder
```

To prepare the corpus:

```bash
# Create unified corpus
python -m src.data_prep.prepare_corpus

# Chunk the corpus
python -m src.data_prep.chunk_corpus

# Create embeddings and FAISS index
python -m src.retriever.embed_and_index
```

## Usage

### Interactive CLI Demo

**RAG Mode** (with retrieval):
```bash
python -m app.cli --mode rag
```

Example:
```
Q> what are confidentiality obligations of the parties?
Retrieving and generating answer…
ANSWER:
The confidentiality obligations of the parties are as follows: each party shall hold the 
other party's Confidential Information in confidence and shall not disclose such Confidential 
Information to third parties nor use the other party's Confidential Information for any 
purpose other than as required to perform under the Agreement...

SOURCES:
[1] cuad | cuad::2ThemartComInc_19990826_10-12G_EX-10.10... | score=0.730
[2] cuad | cuad::QIWI_06_16_2017-EX-99.(D)(2)... | score=0.712
...
```

**Zero-Shot Mode** (without retrieval):
```bash
python -m app.cli --mode zero-shot
```

### Evaluation

**Retrieval Evaluation** (FAISS vs BM25):
```bash
python -m src.evaluation.retrieval_eval
```

**Generation Evaluation** (RAG vs Zero-Shot):
```bash
python -m src.evaluation.generation_eval
```

## Results

### Retrieval Performance (10 queries, k=10)

| Model | Recall@10 | MRR@10 |
|-------|-----------|--------|
| FAISS (Dense) | 1.0 | 1.0 |
| BM25 | 0.0 | 0.0 |

The FAISS dense retriever achieves perfect performance, demonstrating the effectiveness of semantic embeddings over term-frequency baselines for legal text retrieval.

### Generation Performance (10 QA examples)

| Model | ROUGE-L F1 | BERTScore F1 |
|-------|-----------|--------------|
| RAG | 0.230 | 0.857 |
| Zero-Shot | 0.153 | 0.848 |

The RAG system achieves approximately **50% relative improvement in ROUGE-L** and modest but consistent gains in BERTScore compared to the zero-shot baseline. These results demonstrate that retrieving relevant contract and case law passages before generation produces answers more closely aligned with reference solutions, both lexically and semantically.

### Key Findings

- Dense retrieval significantly outperforms BM25 for legal document retrieval
- RAG substantially improves answer quality compared to zero-shot LLMs
- Retrieved context effectively grounds answers and reduces hallucinations
- Citation linkage enables users to verify claims against source documents

