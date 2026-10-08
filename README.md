# did-agent

An LLM pipeline that turns Rust library source code into a searchable, version-aware knowledge base of its public functions, aimed at Decentralized Identity (DID) libraries such as [IOTA Identity](https://github.com/iotaledger/identity).

## Goal

Library APIs change between releases: functions get renamed, moved, or split. Developers (and coding assistants) often answer "how do I do X with this library?" using documentation for the wrong version. This project aims to answer such questions with functions that actually exist in a specific library version, for example:

> *"How do I generate and sign a message using ed25519?"*
>
> → `generate_ed25519_key_pair` and `generate_ed25519_signature` in `sample 2.0.0`,
> but `generate_key_pair` and `generate_signature` in `sample 1.1.0`.

The two sample crates in [`data/`](data/) reproduce exactly this kind of breaking rename, and serve as a small test case before running the pipeline on the full IOTA Identity crate.

To get there, the project:

1. **Extracts** every public function from each library version and has an LLM write a description of it.
2. **Stores** the result twice: as a graph (project → file → function) in Neo4j, and as embeddings of the descriptions in Milvus for semantic search.
3. **Experiments** with LoRA fine-tuning a small code model so it can answer such questions directly, and evaluates how well it does.

## How it works

```
data/<lib>-<version>/   Rust source, one folder per library version
        │
        │  kg_construction.py
        │    tree-sitter: find top-level `pub fn` items
        │    LLM agent: explain → self-review → retry if "poor"
        ▼
output/<lib>-<version>_kg.json
        │
        ├── kg_import.py ──► Neo4j   (:Project)-[:HAS_FILE]->(:File)-[:CONTAINS]->(:Function)
        │                └─► Milvus  all-MiniLM-L6-v2 embeddings of function descriptions
        │
        └── finetune.py ──► LoRA adapter on Qwen2.5-Coder-0.5B-Instruct ──► inference.py (evaluation)
```

The explanation agent ([`agents/function_explanation_agent.py`](agents/function_explanation_agent.py)) is a LangGraph loop: the model explains a function's parameters, return value and behaviour, then grades its own answer, and tries again (up to `max_tries`) if the grade is "poor". The LLM is served by vLLM through its OpenAI-compatible API; Ollama is also supported ([`agents/model.py`](agents/model.py)).

## Project layout

| Path | Purpose |
|---|---|
| [`kg_construction.py`](kg_construction.py) | Step 1: extract and describe functions, write `output/*.json` |
| [`kg_import.py`](kg_import.py) | Step 2: load into Neo4j and Milvus, run a retrieval check |
| [`finetune.py`](finetune.py) | Step 3: LoRA fine-tune, saved to `finetuned_models/` |
| [`inference.py`](inference.py) | Step 4: evaluate the fine-tuned model, write `evaluation_results.json` |
| [`agents/`](agents/) | Model wrapper, tree-sitter function extractor, explanation agent |
| [`data/`](data/) | Input libraries: two sample crates and IOTA Identity |
| [`output/`](output/) | Extracted knowledge-base JSON |
| [`docker/`](docker/) | Docker Compose for vLLM, Neo4j and Milvus |
| [`test/`](test/) | pytest tests (the model tests need a running Ollama) |

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.13, and Docker (or OrbStack on macOS).

```bash
# 1. Install Python dependencies
uv sync

# 2. Start vLLM (:8000), Neo4j (:7474 browser, :7687 bolt) and Milvus (:19530)
docker compose -f docker/docker-compose.yml up -d
docker logs -f vllm            # wait until the model has loaded

# 3. Build the knowledge base
uv run kg_construction.py      # or ./script.sh to run it in the background

# 4. Load it into Neo4j + Milvus and check retrieval
uv run kg_import.py

# 5. Optional: fine-tuning experiment
uv run finetune.py
uv run inference.py
```

Neo4j's browser is at <http://localhost:7474> (user `neo4j`, password `test`).

### Platform notes

- The compose file uses vLLM's **CPU** image for arm64 (`vllm/vllm-openai-cpu:latest-arm64`), so it runs on Apple Silicon without a GPU. `--enforce-eager` is required there, because Torch's kernel compilation fails on Apple M-series CPUs inside the container. On an NVIDIA machine, switch to `vllm/vllm-openai` and drop that flag.
- `vllm` and `bitsandbytes` are installed only on Linux (see `pyproject.toml`); on macOS the LLM runs only in the container.
- Milvus runs standalone with embedded etcd and local-disk storage, so there are no MinIO or etcd containers.

## Status

This is a work in progress.

- **Extraction works** on the sample crates. Only top-level `pub fn` items are extracted, so methods in `impl` blocks are not covered yet, which matters for IOTA Identity.
- **Library selection is hardcoded** in `kg_construction.py` to the two sample crates; IOTA Identity has not been processed yet.
- **Neo4j/Milvus retrieval** is the main path for answering questions.
- **Fine-tuning does not work yet.** It is trained on only 6 keyword-matched examples, and the last evaluation scored 0/6: the model suggests unrelated libraries (`openssl`, `libsecp256`) instead of the project's own functions.
