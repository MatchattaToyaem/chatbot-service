"""
CLI runner for RAG evaluation.

Usage:
    python run_eval.py --retrieval              # Retrieval accuracy only (no LLM)
    python run_eval.py --faithfulness           # Full pipeline + LLM judge
    python run_eval.py --full                   # Both
    python run_eval.py --retrieval --category Procedures   # One category
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.service import RAGService
from eval.questions import EVAL_QUESTIONS, get_questions_by_category, get_categories
from eval.retrieval_eval import RetrievalEvaluator
from eval.faithfulness_eval import FaithfulnessEvaluator

logging.basicConfig(
    level=logging.WARNING,  # Quiet — only show eval output
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def run_retrieval_eval(questions, rag_service, top_k):
    """Run retrieval accuracy evaluation."""
    print("\n" + "=" * 60)
    print("RETRIEVAL ACCURACY EVALUATION")
    print(f"Questions: {len(questions)} | Top-K: {top_k}")
    print("=" * 60 + "\n")

    evaluator = RetrievalEvaluator(rag_service)
    summary = evaluator.evaluate(questions, retrieval_k=20, final_k=top_k)

    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Hit Rate:  {summary.hit_rate:.1%} ({int(summary.hit_rate * summary.total_questions)}/{summary.total_questions})")
    print(f"  MRR:       {summary.mrr:.3f}")

    print(f"\n  Per Category:")
    for cat, stats in sorted(summary.per_category.items()):
        print(f"    {cat:15s}: {stats['hit_rate']:.1%} ({stats['hits']}/{stats['total']})")

    # Show misses
    misses = [r for r in summary.results if not r.hit]
    if misses:
        print(f"\n  MISSES ({len(misses)}):")
        for r in misses:
            print(f"    [{r.category}] {r.question[:60]}")
            print(f"      Top sources: {r.top_sources}")

    return summary


def run_faithfulness_eval(questions, rag_service):
    """Run faithfulness evaluation (requires Ollama)."""
    print("\n" + "=" * 60)
    print("FAITHFULNESS EVALUATION")
    print(f"Questions: {len(questions)} | Judge: llama3.2")
    print("=" * 60 + "\n")

    evaluator = FaithfulnessEvaluator(rag_service)
    summary = evaluator.evaluate(questions)

    print(f"\n{'=' * 60}")
    print(f"FAITHFULNESS RESULTS")
    print(f"{'=' * 60}")
    print(f"  Avg Faithfulness: {summary.avg_faithfulness:.3f}")
    target = 0.80
    status = "PASS" if summary.avg_faithfulness >= target else "FAIL"
    print(f"  Target:           {target:.2f} — {status}")

    print(f"\n  Per Category:")
    for cat, stats in sorted(summary.per_category.items()):
        print(f"    {cat:15s}: {stats['avg_faithfulness']:.3f}")

    # Show low scores
    low = [r for r in summary.results if r.score < 0.5]
    if low:
        print(f"\n  LOW FAITHFULNESS ({len(low)}):")
        for r in low:
            print(f"    [{r.category}] {r.question[:50]} → {r.score:.2f}")
            print(f"      {r.reasoning[:80]}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="O'Connors IMS RAG Evaluation")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval accuracy eval")
    parser.add_argument("--faithfulness", action="store_true", help="Run faithfulness eval (needs Ollama)")
    parser.add_argument("--full", action="store_true", help="Run both evaluations")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval eval")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()

    if not (args.retrieval or args.faithfulness or args.full):
        args.retrieval = True  # Default to retrieval only

    # Filter questions
    if args.category:
        questions = get_questions_by_category(args.category)
        if not questions:
            print(f"No questions found for category '{args.category}'")
            print(f"Available: {get_categories()}")
            sys.exit(1)
    else:
        questions = EVAL_QUESTIONS

    # Initialise RAG service
    print("Initialising RAG service...")
    rag = RAGService()

    results = {}

    if args.retrieval or args.full:
        ret_summary = run_retrieval_eval(questions, rag, args.top_k)
        results["retrieval"] = {
            "hit_rate": ret_summary.hit_rate,
            "mrr": ret_summary.mrr,
            "per_category": ret_summary.per_category,
        }

    if args.faithfulness or args.full:
        faith_summary = run_faithfulness_eval(questions, rag)
        results["faithfulness"] = {
            "avg_faithfulness": faith_summary.avg_faithfulness,
            "per_category": faith_summary.per_category,
        }

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
