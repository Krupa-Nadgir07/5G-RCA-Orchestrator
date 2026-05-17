# Multi-Agent Orchestrator for Automated Root Cause Analysis in 5G gNB

An AI-powered system that performs automated root cause analysis (RCA) for 5G gNodeB network issues using Small Language Models (SLMs), Retrieval-Augmented Generation (RAG), and Chain-of-Thought (CoT) reasoning.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌────────────────────────────────────┐
│  Kafka/API  │────▶│  Ingestion    │────▶│        Multi-Agent Orchestrator    │
│  (Events)   │      │  + Preprocess│      │                                    │
└─────────────┘      └──────────────┘      │  Signal → Knowledge → Hypothesis   │
                                           │           → Validation → Decision  │
                                           └──────────────┬─────────────────────┘
                                                          │
                                           ┌──────────────┼──────────────┐
                                           ▼              ▼              ▼
                                     ┌──────────┐  ┌──────────┐  ┌──────────┐
                                     │  RAG     │  │  SLM     │  │  LLM     │
                                     │Qdrant+ES │  │(vLLM)    │  │(GPT-4o)  │
                                     └──────────┘  └──────────┘  └──────────┘
```

### Multi-Agent Pipeline

1. **Signal Analysis Agent** - Detects KPI anomalies and correlation patterns
2. **Knowledge Retrieval Agent** - Fetches relevant 3GPP/vendor docs via hybrid RAG
3. **Hypothesis Agent** - Generates root cause hypotheses using CoT reasoning
4. **Validation Agent** - Validates hypotheses against evidence
5. **Decision Agent** - Produces final RCA result, escalates to LLM if needed

### Inference Routing

- **Tier 1**: Phi-3-mini (3.8B) - Fast, handles simple cases
- **Tier 2**: Mistral-7B - Mid-tier for moderate complexity
- **Tier 3**: GPT-4o - Complex cases requiring deep reasoning

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- GPU (recommended for local SLM inference)

### 1. Setup

```bash
# Clone and setup
cd Agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode (fixes all internal imports)
pip install -e .
```

### 2. Start Infrastructure

```bash
# Start Qdrant and Redis
docker-compose up -d

# Load knowledge base into Qdrant
python scripts/load_knowledge.py

# Start Ollama with Mistral (required for inference)
ollama pull mistral
ollama serve
```

### 3. Run the API

```bash
uvicorn services.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Or use the entry point:

```bash
python main.py
```

### 4. Run Demo (no API server needed)

```bash
python scripts/run_demo.py
```

### 5. Test

```bash
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Run RCA on an event |
| POST | `/api/v1/analyze/batch` | Batch analysis |
| GET | `/api/v1/rca/{id}` | Get stored RCA result |
| POST | `/api/v1/benchmark` | Run evaluation benchmark |
| POST | `/api/v1/feedback` | Submit operator feedback |
| POST | `/api/v1/knowledge/ingest` | Add documents to RAG |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/metrics` | System metrics |

## Project Structure

```
Agent/
├── config/settings.py          # Configuration (Pydantic Settings)
├── models/schemas.py           # Data models & contracts
├── services/
│   ├── api/                    # FastAPI gateway
│   ├── ingestion/              # Event ingestion & validation
│   ├── preprocessing/          # Log parsing & KPI extraction
│   ├── rag/                    # Hybrid retrieval (dense + sparse)
│   ├── inference/              # SLM/LLM routing & confidence calibration
│   ├── orchestrator/           # Multi-agent graph execution
│   │   ├── agents/            # Individual agent implementations
│   │   ├── graph.py           # Pipeline orchestrator
│   │   └── working_memory.py  # Redis-backed inter-agent memory
│   └── evaluation/            # Benchmarking framework
├── tests/                     # Unit & integration tests
├── deployment/                # Monitoring configs
├── docker-compose.yml         # Full stack deployment
├── Dockerfile                 # Application container
├── requirements.txt           # Python dependencies
└── SYSTEM_DESIGN.md          # Detailed architecture document
```

## Evaluation

Run benchmarks comparing SLM vs LLM performance:

```bash
curl -X POST http://localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"dataset": "synthetic_5g", "max_samples": 50}'
```

Metrics tracked:
- **Classification F1** - Root cause identification accuracy
- **Reasoning Accuracy** - CoT step correctness
- **Latency P95** - End-to-end response time
- **Calibration ECE** - Confidence reliability
- **Hallucination Rate** - Factual correctness
- **Cost per Query** - API/compute costs

## Configuration

Key environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers | `localhost:9092` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379` |
| `MONGODB_URL` | MongoDB connection | `mongodb://localhost:27017` |
| `QDRANT_URL` | Qdrant vector DB | `http://localhost:6333` |
| `VLLM_BASE_URL` | vLLM server | `http://localhost:8001/v1` |
| `OPENAI_API_KEY` | OpenAI key for GPT-4o fallback | - |
| `CONFIDENCE_THRESHOLD` | SLM→LLM escalation threshold | `0.7` |

## License

Research project - Academic use.
