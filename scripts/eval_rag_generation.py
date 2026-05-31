import os
import sys
import json
import time
import asyncio
from pathlib import Path
import re

# Add repo root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.rag.service import RAGService
from services.inference.slm_client import SLMClient
from models.schemas import InferenceRequest
from scripts.benchmark import load_scenarios

# Enable UTF-8 encoding support for print statements
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Prompts for candidate generation
CANDIDATE_SYSTEM_PROMPT = """You are a 5G network expert. You will receive a drive-test scenario and relevant 3GPP context.
Identify the root cause category (C1-C8) and provide a concise (2-3 sentence) reasoning citing evidence from the context."""

CANDIDATE_USER_PROMPT = """## Drive-Test Scenario:
{scenario}

## Retrieved 3GPP Context:
{context}

Please classify and explain."""

# Judge prompt to evaluate RAG Triad
JUDGE_SYSTEM_PROMPT = """You are an AI Quality Evaluator specializing in Retrieval Augmented Generation (RAG) validation.
You will be given the original User Query/Scenario, the Retrieved Context (retrieved by RAG), and the Generated Response (by the candidate model).
Evaluate the performance based on the following three criteria on a scale of 0.0 to 1.0:

1. Faithfulness: Is the Generated Response fully grounded in the Retrieved Context? (1.0 = all claims are backed by context; 0.0 = contains hallucinations or external knowledge).
2. Context Relevance: How relevant is the Retrieved Context to the User Query? (1.0 = context directly explains the query's drive-test anomalies; 0.0 = context is generic or completely irrelevant).
3. Answer Relevance: Does the Generated Response directly and accurately answer the User Query? (1.0 = addresses the query completely and correctly; 0.0 = irrelevant or incorrect classification).

Output a JSON object with EXACTLY this structure, containing no other text:
{
  "faithfulness": {
    "score": <0.0-1.0>,
    "reasoning": "<1 sentence justification>"
  },
  "context_relevance": {
    "score": <0.0-1.0>,
    "reasoning": "<1 sentence justification>"
  },
  "answer_relevance": {
    "score": <0.0-1.0>,
    "reasoning": "<1 sentence justification>"
  }
}"""

JUDGE_USER_PROMPT = """## Original Query/Scenario:
{query}

## Retrieved Context:
{context}

## Generated Response:
{response}

Evaluate the triad quality and return JSON."""

async def main():
    n_scenarios = 25 # 5 scenarios for a faster, cost-effective run
    candidate_model = "llama-3.1-8b-instant"  # Model under test
    judge_model = "llama-3.3-70b-versatile"    # Evaluator model

    print("\n" + "═" * 70)
    print("  5G RCA RAG Generation Evaluation (LLM-as-a-Judge RAG Triad)")
    print(f"  Candidate: {candidate_model} | Judge: {judge_model}")
    print("═" * 70)

    # Initialize services
    print("📂 Initializing RAG Service and loading embedding models...")
    rag = RAGService()
    await rag.initialize()

    candidate_client = SLMClient(model_name=candidate_model)
    judge_client = SLMClient(model_name=judge_model)

    # Load test scenarios
    dataset_path = Path("data/rag_dataset_2.jsonl")
    try:
        scenarios = load_scenarios(str(dataset_path), n=n_scenarios, seed=42)
    except FileNotFoundError:
        print(f"Error: evaluation dataset not found at {dataset_path}")
        return

    scores = {
        "faithfulness": 0.0,
        "context_relevance": 0.0,
        "answer_relevance": 0.0,
    }
    eval_count = 0

    print(f"\nRunning end-to-end evaluation on {len(scenarios)} scenarios...\n")

    for i, scenario in enumerate(scenarios):
        meta = scenario.get("metadata", {})
        true_label = meta["answer"]
        rc_cat = meta.get("root_cause_category", "")
        content = scenario.get("content", "")
        scenario_id = scenario.get("scenario_id", f"scenario_{i}")

        print(f"  [{i+1:02d}/{len(scenarios)}] Processing {scenario_id} (True={true_label})...")

        # Step 1: Retrieve context
        rag_query = f"5G drive test root cause analysis: {rc_cat}"
        result = await rag.retrieve(rag_query, top_k=4)
        context_str = rag.format_context(result, max_tokens=1500) or "No context retrieved."

        # Step 2: Candidate generates response
        cand_prompt = CANDIDATE_USER_PROMPT.format(scenario=content[:2000], context=context_str)
        cand_req = InferenceRequest(
            prompt=cand_prompt,
            system_prompt=CANDIDATE_SYSTEM_PROMPT,
            max_tokens=256,
            temperature=0.1
        )
        cand_res = await candidate_client.generate(cand_req)
        response_text = cand_res.text or ""

        # Step 3: Judge evaluates the generation
        judge_prompt = JUDGE_USER_PROMPT.format(
            query=content[:1500],
            context=context_str[:2000],
            response=response_text
        )
        judge_req = InferenceRequest(
            prompt=judge_prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.0
        )
        
        # Enforce delay to avoid Groq rate limits
        await asyncio.sleep(1.5)
        
        judge_res = await judge_client.generate(judge_req)
        judge_raw = judge_res.text or ""

        # Parse judge JSON
        try:
            # Extract JSON block using first '{' and last '}'
            start_idx = judge_raw.find("{")
            end_idx = judge_raw.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = judge_raw[start_idx:end_idx+1]
                eval_data = json.loads(json_str)
            else:
                eval_data = json.loads(judge_raw)

            f_score = float(eval_data["faithfulness"]["score"])
            c_score = float(eval_data["context_relevance"]["score"])
            a_score = float(eval_data["answer_relevance"]["score"])

            scores["faithfulness"] += f_score
            scores["context_relevance"] += c_score
            scores["answer_relevance"] += a_score
            eval_count += 1

            print(f"      ↳ Faithfulness: {f_score:.2f} | Context Relevance: {c_score:.2f} | Answer Relevance: {a_score:.2f}")
        except Exception as e:
            print(f"      ✗ Error parsing judge response: {e}")
            print(f"        Raw Judge Output: {judge_raw[:400]}...")

        # Enforce rate limit delay
        await asyncio.sleep(1.5)

    if eval_count > 0:
        avg_f = scores["faithfulness"] / eval_count
        avg_c = scores["context_relevance"] / eval_count
        avg_a = scores["answer_relevance"] / eval_count

        print("\n" + "═" * 70)
        print("                 RAG TRIAD GENERATION REPORT")
        print("═" * 70)
        print(f"Evaluated scenarios: {eval_count}")
        print(f"  • Average Faithfulness (Groundedness): {avg_f:.2%}")
        print(f"  • Average Context Relevance:           {avg_c:.2%}")
        print(f"  • Average Answer Relevance:            {avg_a:.2%}")
        print("═" * 70 + "\n")
    else:
        print("\nError: No evaluations completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
