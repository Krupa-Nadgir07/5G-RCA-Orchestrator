# 5G gNB RCA Multi-Agent System — Repo Recovery & Build Blueprint

## Objective

You already have:
- Docker containers
- Ollama installed
- Mistral model downloaded
- Base repository structure

The repository is not completely wired together yet. The goal is NOT to immediately build a production telecom platform.

The goal is:
1. Make the current architecture actually executable.
2. Run a complete end-to-end RCA flow.
3. Use a dummy dataset first.
4. Preserve the research architecture from the system design document.
5. Create a journal-paper-grade prototype with explainable multi-agent RCA.

This document gives you a complete recovery plan from the repo's current broken state.

---

# 1. Current Repository Health Assessment

## What Is Already Good

The repo structure is actually strong.

You already have:

- FastAPI gateway
- Multi-agent orchestration
- RAG layer
- Inference abstraction layer
- Evaluation framework
- Docker infrastructure
- Kafka + Redis + Mongo + Qdrant stack
- Typed Pydantic schemas
- Working project structure
- Research-grade architecture documents

This is NOT a bad repo.

It is mostly an incomplete integration problem.

---

# 2. Main Problems Preventing Execution

## Problem 1 — Python Module Resolution

Tests fail because PYTHONPATH is not configured.

Example failure:

```bash
ModuleNotFoundError: No module named 'models'
```

### Fix

Always run the project using:

```bash
PYTHONPATH=.
```

Example:

```bash
PYTHONPATH=. uvicorn services.api.app:app --reload
```

OR add this to `.env`:

```env
PYTHONPATH=.
```

OR install the package in editable mode later.

---

## Problem 2 — Ollama Not Connected

The repo currently assumes:

- vLLM
- OpenAI fallback

But your actual local setup is:

- Ollama
- Mistral

So the inference layer must be rewired.

---

## Problem 3 — RAG Pipeline Is Not Fully Operational

The architecture exists.

But:
- no actual embeddings pipeline is fully wired
- no indexed documents
- no telecom corpus
- no ingestion bootstrap

---

## Problem 4 — No Minimal End-to-End Dataset

The system architecture assumes:

- live gNB logs
- streaming KPIs
- telecom event pipelines

But for now you need:

- small synthetic RCA dataset
- deterministic outputs
- reproducible experiments
- paper-ready benchmarks

---

## Problem 5 — Multi-Agent Loop Exists but Is Not Operationally Bound

The orchestrator is architecturally correct.

But:
- agents are not strongly connected to actual retrieval outputs
- confidence propagation is weak
- RCA scoring is not calibrated
- memory graph is incomplete

---

# 3. What You SHOULD Build First

DO NOT build:

- full telecom streaming infra
- production Kubernetes
- GPU autoscaling
- live Kafka ingestion
- large vector DB

FIRST build:

# Minimal Research Prototype (MRP)

This is what your journal paper actually needs.

The paper needs:

- a working architecture
- reproducible RCA flow
- explainable reasoning
- comparison between agents/models
- measurable accuracy
- retrieval augmentation

NOT carrier-grade infrastructure.

---

# 4. Target Minimal Working Architecture

You should simplify the stack to this:

```text
Dummy Dataset
    ↓
Preprocessor
    ↓
Signal Agent
    ↓
RAG Retrieval
    ↓
Hypothesis Agent
    ↓
Validation Agent
    ↓
Decision Agent
    ↓
JSON RCA Output
```

And use:

```text
Ollama + Mistral
```

for ALL inference initially.

Avoid OpenAI until later.

---

# 5. Recommended Final Minimal Stack

## Keep

- FastAPI
- Ollama
- Mistral
- Qdrant
- Redis
- Multi-agent orchestration
- LangGraph-style flow

## Temporarily Remove

- Kafka
- Elasticsearch
- OpenAI dependency
- vLLM
- Prometheus/Grafana
- complex async consumers

This massively reduces instability.

---

# 6. Immediate Recovery Plan

# PHASE 1 — Make the Repo Runnable

## Step 1 — Create Clean Python Environment

```bash
cd Agent
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then install missing packages:

```bash
pip install structlog sentence-transformers langchain-community
```

---

## Step 2 — Create Minimal `.env`

Use:

```env
APP_ENV=development
APP_DEBUG=true

REDIS_URL=redis://localhost:6379/0
MONGODB_URL=mongodb://localhost:27017

QDRANT_HOST=localhost
QDRANT_PORT=6333

OLLAMA_BASE_URL=http://localhost:11434
SLM_MODEL=mistral

PYTHONPATH=.
```

---

## Step 3 — Start ONLY Required Containers

Use:

```bash
docker compose up -d redis mongodb qdrant
```

DO NOT start Kafka yet.

---

## Step 4 — Verify Ollama

```bash
ollama run mistral
```

Then:

```bash
curl http://localhost:11434/api/tags
```

---

# 7. Critical Code Refactor — Replace vLLM with Ollama

This is the SINGLE MOST IMPORTANT FIX.

---

# File to Modify

```text
services/inference/slm_client.py
```

Replace the current inference logic with Ollama.

---

# Recommended Ollama Client

```python
import httpx

class SLMClient:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "mistral"

    async def generate(self, prompt: str):
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
            )

            data = response.json()
            return data["response"]
```

This alone will make the pipeline functional.

---

# 8. Build Dummy RCA Dataset

You MUST do this before touching real logs.

---

# Recommended Dataset Structure

Create:

```text
data/dummy_rca_dataset.json
```

Example:

```json
[
  {
    "event_id": "evt_001",
    "cell_id": "cell_101",
    "issue": "throughput_degradation",
    "kpis": {
      "sinr_db": 3,
      "prb_util": 96,
      "bler": 18,
      "handover_fail_rate": 2
    },
    "ground_truth": "interference"
  },
  {
    "event_id": "evt_002",
    "cell_id": "cell_205",
    "issue": "handover_failure",
    "kpis": {
      "sinr_db": 18,
      "prb_util": 40,
      "bler": 2,
      "handover_fail_rate": 35
    },
    "ground_truth": "neighbor_relation_issue"
  }
]
```

Start with:

- 25 samples
- 4 RCA classes
- deterministic KPI patterns

---

# 9. Minimal RCA Categories for Journal Prototype

Use only these initially:

| RCA Category | KPI Signature |
|---|---|
| Interference | Low SINR + High BLER |
| Congestion | High PRB Utilization |
| Handover Failure | HO Fail Rate High |
| Hardware Fault | Random KPI instability |

This is enough for a first paper.

---

# 10. Simplify the Multi-Agent Flow

Your current architecture is too ambitious for the first execution.

Implement this simplified flow:

---

## Agent 1 — Signal Agent

Input:

```json
KPIs
```

Output:

```json
Detected anomalies
```

Use simple threshold rules.

Example:

```python
if sinr < 5:
    anomalies.append("low_sinr")
```

DO NOT use LLMs here.

---

## Agent 2 — Knowledge Agent

Input:

```text
Detected anomalies
```

Retrieves:

```text
Relevant telecom troubleshooting docs
```

Initially use:

```text
local JSON documents
```

instead of full 3GPP corpus.

---

## Agent 3 — Hypothesis Agent

THIS is where Mistral matters.

Prompt example:

```text
Given:
- low SINR
- high BLER
- high interference power

Generate the most likely root cause.
```

Expected output:

```text
Likely root cause: RF interference.
```

---

## Agent 4 — Validation Agent

Use rule-based scoring.

Example:

```python
confidence = 0.85
```

based on:

- KPI alignment
- retrieved docs
- hypothesis consistency

---

## Agent 5 — Decision Agent

Returns final JSON:

```json
{
  "root_cause": "interference",
  "confidence": 0.89,
  "reasoning": [
    "SINR below threshold",
    "BLER elevated",
    "RAG retrieved interference document"
  ]
}
```

This is already publishable architecture material.

---

# 11. RAG Recovery Plan

# DO NOT START WITH ELASTICSEARCH

Start with:

```text
SentenceTransformer embeddings + Qdrant only
```

---

# Recommended Embedding Model

Use:

```text
BAAI/bge-small-en-v1.5
```

Very good for lightweight RAG.

---

# Minimal Knowledge Base

Create:

```text
knowledge/
```

with:

```text
interference.txt
congestion.txt
handover.txt
hardware_fault.txt
```

Each file should contain:

- symptoms
- KPIs
- remediation
- telecom notes

Example:

```text
Interference is associated with:
- low SINR
- elevated BLER
- unstable throughput
```

This is enough for the first experiment.

---

# 12. Qdrant Initialization

Create a startup ingestion script:

```text
scripts/load_knowledge.py
```

Pipeline:

```text
TXT → chunks → embeddings → Qdrant
```

This gives you:

- true RAG
- vector retrieval
- explainability
- research credibility

---

# 13. Journal Paper Strategy

This is VERY IMPORTANT.

Your paper is NOT about production deployment.

Your paper is about:

```text
Multi-Agent Explainable RCA using SLM + RAG
```

The novelty is:

- modular agent reasoning
- explainable RCA chain
- lightweight SLM deployment
- telecom-domain retrieval augmentation
- confidence-calibrated diagnosis

That is enough for:

- IEEE conference
- Springer chapter
- Elsevier telecom AI paper
- ICC/Globecom workshop

if experiments are clean.

---

# 14. What Experiments You Should Run

## Experiment 1 — SLM vs No-RAG

Measure:

- RCA accuracy
- hallucination reduction
- confidence quality

---

## Experiment 2 — Single-Agent vs Multi-Agent

Measure:

- reasoning quality
- interpretability
- latency

---

## Experiment 3 — Phi-3 vs Mistral

Measure:

- latency
- token cost
- RCA precision

---

## Experiment 4 — Confidence Calibration

Measure:

- confidence vs correctness

This becomes a strong paper section.

---

# 15. Recommended Final Directory Structure

```text
Agent/
├── data/
│   └── dummy_rca_dataset.json
│
├── knowledge/
│   ├── interference.txt
│   ├── congestion.txt
│   ├── handover.txt
│   └── hardware_fault.txt
│
├── scripts/
│   └── load_knowledge.py
│
├── services/
│   ├── orchestrator/
│   ├── rag/
│   ├── inference/
│   └── preprocessing/
│
└── notebooks/
    └── evaluation.ipynb
```

---

# 16. The FASTEST Path to a Working Demo

If your goal is:

```text
"I need a working research demo quickly"
```

Then do this in order:

1. Fix imports
2. Replace vLLM with Ollama
3. Create dummy dataset
4. Create 4 knowledge documents
5. Enable Qdrant embeddings
6. Run orchestrator locally
7. Expose FastAPI endpoint
8. Save RCA JSON outputs
9. Build evaluation notebook
10. Generate plots for paper

This gets you a demonstrable system very quickly.

---

# 17. Biggest Architectural Advice

Your current repo is trying to behave like:

```text
carrier-grade telecom AI platform
```

But your actual immediate need is:

```text
research-grade RCA demonstrator
```

Those are VERY different engineering targets.

Simplify aggressively first.

Then scale later.

---

# 18. Immediate Next Actions (Most Important)

Do THESE first:

## Priority 1

- Make Ollama inference work
- Make orchestrator execute sequentially
- Produce RCA JSON output

## Priority 2

- Add Qdrant retrieval
- Add synthetic dataset
- Add evaluation metrics

## Priority 3

- Add dashboards
- Add Kafka
- Add monitoring
- Add OpenAI fallback

---

# 19. Final Recommended Architecture for Version 1

```text
FastAPI
   ↓
Orchestrator
   ↓
Signal Agent
   ↓
Qdrant RAG
   ↓
Mistral via Ollama
   ↓
Validation Logic
   ↓
JSON RCA Output
```

This architecture is:

- achievable
- stable
- explainable
- publishable
- experimentally measurable

---

# 20. Final Guidance

You are MUCH closer than you think.

The repo already contains:

- the architecture
- the abstractions
- the orchestration concepts
- the service boundaries
- the evaluation framework

What is missing is:

- simplification
- operational wiring
- minimal datasets
- local inference integration

Do NOT attempt full telecom production infrastructure yet.

Build the minimal explainable RCA pipeline first.

Once that works:

- replace dummy logs
- expand RAG corpus
- add streaming
- add larger datasets
- benchmark models
- improve retrieval
- optimize agents

That becomes your PhD-grade or publication-grade evolution path.

