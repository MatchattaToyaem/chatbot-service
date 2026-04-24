"""
Retrieval accuracy evaluation for the RAG pipeline.

Measures whether the correct source documents appear in the top-k
retrieval results for each ground truth question. Does not require
an LLM — tests the embedding + reranking pipeline only.

Metrics:
    - Hit rate: % of questions where at least one expected source is in top-k
    - Mean Reciprocal Rank (MRR): average of 1/rank of first correct source
    - Per-category breakdown
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result for a single evaluation question."""
    question: str
    category: str
    hit: bool
    first_hit_rank: int  # 0 = not found
    matched_sources: list[str]
    top_sources: list[str]


@dataclass
class RetrievalEvalSummary:
    """Summary metrics for the full evaluation."""
    total_questions: int
    hit_rate: float
    mrr: float
    per_category: dict
    results: list[RetrievalResult]


class RetrievalEvaluator:
    """
    Evaluates retrieval accuracy against ground truth questions.

    For each question, checks if any of the expected source file patterns
    appear in the top-k retrieved results. A "hit" means at least one
    expected source is found.
    """

    def __init__(self, rag_service):
        self._rag = rag_service

    def _check_hit(
        self, expected_sources: list[str], retrieved_sources: list[str]
    ) -> tuple[bool, int, list[str]]:
        """Check if any expected source pattern matches any retrieved source."""
        matched = []
        first_rank = 0

        for rank, source in enumerate(retrieved_sources, 1):
            source_lower = source.lower()
            for expected in expected_sources:
                if expected.lower() in source_lower:
                    if not matched:
                        first_rank = rank
                    matched.append(f"{expected} (rank {rank})")

        return len(matched) > 0, first_rank, matched

    def evaluate(
        self,
        questions: list[dict],
        retrieval_k: int = 20,
        final_k: int = 5,
    ) -> RetrievalEvalSummary:
        """
        Run retrieval evaluation on a set of ground truth questions.

        Args:
            questions: List of question dicts from eval.questions.
            retrieval_k: Initial ChromaDB retrieval count.
            final_k: Final count after reranking.

        Returns:
            RetrievalEvalSummary with hit rate, MRR, and per-category breakdown.
        """
        results = []
        category_stats = {}

        for i, q in enumerate(questions):
            question = q["question"]
            expected = q["expected_sources"]
            category = q["category"]

            sys.stdout.write(f"  [{i+1}/{len(questions)}] {question[:60]}...")
            sys.stdout.flush()

            # Retrieve and rerank
            reranked = self._rag.retrieve(
                question=question,
                retrieval_k=retrieval_k,
                final_k=final_k,
            )

            # Extract source files from results
            top_sources = [
                r.metadata.get("source_file", "?") for r in reranked
            ]

            # Check hit
            hit, first_rank, matched = self._check_hit(expected, top_sources)

            result = RetrievalResult(
                question=question,
                category=category,
                hit=hit,
                first_hit_rank=first_rank,
                matched_sources=matched,
                top_sources=top_sources[:3],
            )
            results.append(result)

            # Track per-category
            if category not in category_stats:
                category_stats[category] = {"hits": 0, "total": 0, "mrr_sum": 0.0}
            category_stats[category]["total"] += 1
            if hit:
                category_stats[category]["hits"] += 1
                category_stats[category]["mrr_sum"] += 1.0 / first_rank

            status = "HIT" if hit else "MISS"
            print(f" {status}")

        # Compute summary
        total = len(results)
        hits = sum(1 for r in results if r.hit)
        hit_rate = hits / total if total > 0 else 0.0

        mrr_sum = sum(1.0 / r.first_hit_rank for r in results if r.hit)
        mrr = mrr_sum / total if total > 0 else 0.0

        per_category = {}
        for cat, stats in category_stats.items():
            cat_rate = stats["hits"] / stats["total"] if stats["total"] > 0 else 0.0
            cat_mrr = stats["mrr_sum"] / stats["total"] if stats["total"] > 0 else 0.0
            per_category[cat] = {
                "hit_rate": round(cat_rate, 3),
                "mrr": round(cat_mrr, 3),
                "hits": stats["hits"],
                "total": stats["total"],
            }

        return RetrievalEvalSummary(
            total_questions=total,
            hit_rate=round(hit_rate, 3),
            mrr=round(mrr, 3),
            per_category=per_category,
            results=results,
        )
