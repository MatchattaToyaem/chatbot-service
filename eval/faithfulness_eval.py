"""
Faithfulness evaluation for generated answers.

Uses an LLM (Ollama) as a judge to score whether the generated answer
is faithfully grounded in the retrieved context. Each claim in the
answer is checked against the source chunks.

Scoring:
    1.0 = fully faithful (every claim supported by context)
    0.5 = partially faithful (some claims unsupported)
    0.0 = unfaithful (hallucinated or contradicts context)
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are evaluating the faithfulness of an AI-generated answer.

CONTEXT (retrieved documents):
{context}

QUESTION: {question}

ANSWER: {answer}

Score the answer's faithfulness on this scale:
- 1.0: Every claim in the answer is directly supported by the context documents.
- 0.75: Most claims are supported, with minor unsupported details.
- 0.5: Some claims are supported, but significant information is not in the context.
- 0.25: Few claims are supported; mostly unsupported or vague.
- 0.0: The answer contradicts the context or is entirely hallucinated.

Respond with ONLY a JSON object:
{{"score": <float>, "reasoning": "<one sentence explanation>"}}"""


@dataclass
class FaithfulnessResult:
    """Faithfulness score for a single question-answer pair."""
    question: str
    category: str
    score: float
    reasoning: str
    answer_preview: str


@dataclass
class FaithfulnessEvalSummary:
    """Summary of faithfulness evaluation."""
    total_questions: int
    avg_faithfulness: float
    per_category: dict
    results: list[FaithfulnessResult]


class FaithfulnessEvaluator:
    """
    Evaluates faithfulness of generated answers using an LLM judge.

    Runs the full RAG pipeline (retrieve → rerank → generate) for each
    question, then asks a separate LLM call to judge whether the answer
    is grounded in the retrieved context.
    """

    def __init__(
        self,
        rag_service,
        judge_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
    ):
        self._rag = rag_service
        self._judge_model = judge_model
        self._ollama_url = ollama_url.rstrip("/")

    def _judge_faithfulness(
        self, question: str, context: str, answer: str
    ) -> tuple[float, str]:
        """Ask the LLM judge to score faithfulness."""
        prompt = JUDGE_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )

        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._judge_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 200},
                },
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")

            # Parse JSON from response
            import json
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                score = float(data.get("score", 0.0))
                reasoning = data.get("reasoning", "No reasoning provided.")
                return min(1.0, max(0.0, score)), reasoning

            return 0.0, f"Could not parse judge response: {text[:100]}"

        except Exception as e:
            logger.error("Judge call failed: %s", e)
            return 0.0, f"Judge error: {e}"

    def evaluate(self, questions: list[dict]) -> FaithfulnessEvalSummary:
        """
        Run full faithfulness evaluation.

        For each question:
            1. Run RAG pipeline (retrieve → rerank → generate)
            2. Judge the answer against the retrieved context
            3. Record the faithfulness score

        Args:
            questions: List of question dicts from eval.questions.

        Returns:
            FaithfulnessEvalSummary with average score and per-category breakdown.
        """
        results = []
        category_stats = {}

        for i, q in enumerate(questions):
            question = q["question"]
            category = q["category"]

            sys.stdout.write(f"  [{i+1}/{len(questions)}] {question[:55]}...")
            sys.stdout.flush()

            # Run full RAG pipeline
            response = self._rag.ask(question)

            # Build context string for judge
            context_parts = []
            for s in response.sources:
                context_parts.append(f"[{s['file']}]: (score {s['score']})")
            context_str = "\n".join(context_parts)

            # Judge faithfulness
            score, reasoning = self._judge_faithfulness(
                question=question,
                context=response.answer,  # Judge sees the full answer + sources
                answer=response.answer,
            )

            result = FaithfulnessResult(
                question=question,
                category=category,
                score=score,
                reasoning=reasoning,
                answer_preview=response.answer[:150],
            )
            results.append(result)

            # Track per-category
            if category not in category_stats:
                category_stats[category] = {"scores": [], "total": 0}
            category_stats[category]["scores"].append(score)
            category_stats[category]["total"] += 1

            print(f" {score:.2f} — {reasoning[:60]}")

        # Compute summary
        total = len(results)
        all_scores = [r.score for r in results]
        avg = sum(all_scores) / total if total > 0 else 0.0

        per_category = {}
        for cat, stats in category_stats.items():
            cat_avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
            per_category[cat] = {
                "avg_faithfulness": round(cat_avg, 3),
                "total": stats["total"],
            }

        return FaithfulnessEvalSummary(
            total_questions=total,
            avg_faithfulness=round(avg, 3),
            per_category=per_category,
            results=results,
        )
