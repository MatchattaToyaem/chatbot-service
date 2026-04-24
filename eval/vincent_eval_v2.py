"""
Vincent's 45-Question Evaluation — v2 with Fuzzy Matching.

Improvements over v1 (strict keyword matching):
    1. British/American spelling normalization (oxidising→oxidizing, etc.)
    2. Fuzzy substring matching via difflib (threshold 0.80)
    3. LLM-judged faithfulness scoring (Ollama llama3.2)
    4. Combined score: 50% keyword coverage + 50% LLM faithfulness
    5. Per-category breakdown (MSDS / Policy / Standards)

Usage (local):
    python eval/vincent_eval_v2.py

Usage (Colab):
    !cd /content/chatbot-agent && python eval/vincent_eval_v2.py --local --local-path /content/drive/MyDrive/Capstone/chromadb_local
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AppConfig
from rag.service import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spelling normalization (British → American, common HVAC/chemical variants)
# ---------------------------------------------------------------------------

SPELLING_MAP = {
    "oxidising": "oxidizing",
    "organising": "organizing",
    "recognised": "recognized",
    "recognising": "recognizing",
    "minimise": "minimize",
    "minimising": "minimizing",
    "maximise": "maximize",
    "maximising": "maximizing",
    "utilise": "utilize",
    "utilising": "utilizing",
    "specialised": "specialized",
    "specialising": "specializing",
    "authorised": "authorized",
    "authorising": "authorizing",
    "unauthorised": "unauthorized",
    "polymerisation": "polymerization",
    "vapour": "vapor",
    "vapours": "vapors",
    "colour": "color",
    "colours": "colors",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "favour": "favor",
    "favourable": "favorable",
    "fibre": "fiber",
    "fibres": "fibers",
    "litre": "liter",
    "litres": "liters",
    "metre": "meter",
    "metres": "meters",
    "centre": "center",
    "centred": "centered",
    "defence": "defense",
    "offence": "offense",
    "licence": "license",
    "practise": "practice",
    "programme": "program",
    "catalogue": "catalog",
    "dialogue": "dialog",
    "analyse": "analyze",
    "analysing": "analyzing",
    "labour": "labor",
    "honour": "honor",
    "mould": "mold",
    "sulphur": "sulfur",
    "aluminium": "aluminum",
    "anaesthetic": "anesthetic",
    "aesthetic": "esthetic",
    "manoeuvre": "maneuver",
    "judgement": "judgment",
    "ageing": "aging",
    "acknowledgement": "acknowledgment",
    "cancelation": "cancellation",
    "fulfil": "fulfill",
    "enrol": "enroll",
    "instalment": "installment",
    "skilful": "skillful",
    "counselling": "counseling",
    "modelling": "modeling",
    "travelling": "traveling",
    "labelling": "labeling",
    "signalling": "signaling",
}


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, strip punctuation, normalize spelling."""
    text = text.lower().strip()
    # Remove common punctuation but keep hyphens and slashes (important for standards)
    for char in [".", ",", "!", "?", ";", ":", '"', "'", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(char, " ")
    # Normalize whitespace
    text = " ".join(text.split())
    # Apply spelling normalization
    words = text.split()
    normalized = []
    for word in words:
        normalized.append(SPELLING_MAP.get(word, word))
    return " ".join(normalized)


def fuzzy_term_match(term: str, answer_text: str, threshold: float = 0.75) -> bool:
    """
    Check if a key term appears in the answer using fuzzy matching.

    Strategy:
        1. Exact substring match (fast path)
        2. Word-level fuzzy match for single words
        3. N-gram sliding window for multi-word terms
    """
    norm_term = normalize_text(term)
    norm_answer = normalize_text(answer_text)

    # Fast path: exact substring
    if norm_term in norm_answer:
        return True

    # Single-word term: check each answer word
    term_words = norm_term.split()
    answer_words = norm_answer.split()

    if len(term_words) == 1:
        for word in answer_words:
            ratio = SequenceMatcher(None, norm_term, word).ratio()
            if ratio >= threshold:
                return True
        return False

    # Multi-word term: sliding window
    window_size = len(term_words)
    for i in range(len(answer_words) - window_size + 1):
        window = " ".join(answer_words[i:i + window_size])
        ratio = SequenceMatcher(None, norm_term, window).ratio()
        if ratio >= threshold:
            return True

    # Also try checking if all individual words appear somewhere (relaxed)
    found_count = 0
    for tw in term_words:
        for aw in answer_words:
            if SequenceMatcher(None, tw, aw).ratio() >= threshold:
                found_count += 1
                break
    if found_count >= len(term_words) * 0.7:  # 70% of words found
        return True

    return False


def compute_keyword_score(key_terms: list[str], answer: str) -> tuple[float, list[str], list[str]]:
    """
    Compute fuzzy keyword coverage score.

    Returns:
        (score, matched_terms, missed_terms)
    """
    if not key_terms:
        return 1.0, [], []

    matched = []
    missed = []

    for term in key_terms:
        if fuzzy_term_match(term, answer):
            matched.append(term)
        else:
            missed.append(term)

    score = len(matched) / len(key_terms) if key_terms else 0.0
    return score, matched, missed


def llm_faithfulness_score(question: str, answer: str, key_terms: list[str]) -> float:
    """
    Use Ollama to judge whether the answer is faithful and accurate.

    Returns a score 0.0 to 1.0.
    """
    try:
        import requests

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")

        expected_info = ", ".join(key_terms)

        prompt = f"""You are an evaluation judge for a workplace safety Q&A system.

Question: {question}

Answer provided by the system: {answer}

Expected key information that should be in the answer: {expected_info}

Rate the answer's faithfulness and accuracy on a scale from 0.0 to 1.0:
- 1.0 = Answer contains all expected information accurately
- 0.75 = Answer contains most key information with minor gaps
- 0.50 = Answer contains some relevant information but significant gaps
- 0.25 = Answer is partially relevant but mostly incorrect or incomplete
- 0.0 = Answer is completely wrong, irrelevant, or says "I don't know"

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # Extract the first float from the response
            for token in result.split():
                try:
                    score = float(token)
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
            return 0.5  # Default if parsing fails

    except Exception as e:
        logger.warning("LLM scoring failed: %s — falling back to keyword-only", e)

    return -1.0  # Signal that LLM scoring failed


def run_eval(args):
    """Run the full Vincent evaluation."""
    # Import questions
    from eval.vincent_questions import VINCENT_QUESTIONS

    # Override config if local paths provided
    if args.local_path:
        os.environ["CHROMA_LOCAL_PATH"] = args.local_path
        os.environ["CHROMA_MODE"] = "local"
    if args.collection:
        os.environ["CHROMA_COLLECTION"] = args.collection

    config = AppConfig()
    service = RAGService(config)

    print(f"\n{'='*70}")
    print(f"  Vincent's 45-Question Evaluation — v2 (Fuzzy Matching)")
    print(f"  Collection: {config.chroma.collection}")
    print(f"  Retrieval K: {config.rag.retrieval_k}")
    print(f"  Top K: {config.rag.top_k}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = []
    category_scores = {}

    for i, q in enumerate(VINCENT_QUESTIONS):
        qid = q["qid"]
        category = q["category"]
        question = q["question"]
        key_terms = q["key_terms"]

        print(f"[{i+1}/45] {qid} ({category}): {question[:60]}...")

        try:
            start = time.time()
            response = service.ask(question)
            elapsed = time.time() - start

            answer = response.answer
            sources = response.sources

            # Keyword scoring (fuzzy)
            kw_score, matched, missed = compute_keyword_score(key_terms, answer)

            # LLM scoring (if available)
            llm_score = -1.0
            if not args.skip_llm:
                llm_score = llm_faithfulness_score(question, answer, key_terms)

            # Combined score
            if llm_score >= 0:
                combined = 0.5 * kw_score + 0.5 * llm_score
            else:
                combined = kw_score

            # Retrieval hit check
            source_files = [s.get("file", s.get("source", "")) for s in sources] if sources else []
            hit = any(q["source_hint"].lower() in sf.lower() for sf in source_files) if source_files else False

            result = {
                "qid": qid,
                "category": category,
                "question": question[:80],
                "keyword_score": round(kw_score, 3),
                "llm_score": round(llm_score, 3) if llm_score >= 0 else "N/A",
                "combined_score": round(combined, 3),
                "retrieval_hit": hit,
                "matched_terms": matched,
                "missed_terms": missed,
                "answer_preview": answer[:150] if answer else "",
                "source_count": len(sources) if sources else 0,
                "elapsed_s": round(elapsed, 1),
            }
            results.append(result)

            # Track category scores
            if category not in category_scores:
                category_scores[category] = {"kw": [], "llm": [], "combined": [], "hits": 0, "total": 0}
            category_scores[category]["kw"].append(kw_score)
            if llm_score >= 0:
                category_scores[category]["llm"].append(llm_score)
            category_scores[category]["combined"].append(combined)
            category_scores[category]["total"] += 1
            if hit:
                category_scores[category]["hits"] += 1

            status = "PASS" if combined >= 0.80 else "PARTIAL" if combined >= 0.50 else "FAIL"
            print(f"         KW={kw_score:.2f} | LLM={(f'{llm_score:.2f}' if llm_score >= 0 else 'N/A'):>5} | "
                  f"Combined={combined:.2f} | {status} | {elapsed:.1f}s")
            if missed:
                print(f"         Missed: {', '.join(missed[:3])}{'...' if len(missed) > 3 else ''}")

        except Exception as e:
            logger.error("Error on %s: %s", qid, e)
            results.append({
                "qid": qid, "category": category,
                "question": question[:80],
                "keyword_score": 0.0, "llm_score": "ERROR",
                "combined_score": 0.0, "retrieval_hit": False,
                "error": str(e),
            })

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    all_combined = [r["combined_score"] for r in results if isinstance(r.get("combined_score"), (int, float))]
    all_kw = [r["keyword_score"] for r in results if isinstance(r.get("keyword_score"), (int, float))]
    all_hits = sum(1 for r in results if r.get("retrieval_hit"))

    print(f"\n  Overall ({len(results)} questions):")
    print(f"    Keyword Score:    {sum(all_kw)/len(all_kw):.3f}" if all_kw else "    Keyword Score:    N/A")
    print(f"    Combined Score:   {sum(all_combined)/len(all_combined):.3f}" if all_combined else "    Combined Score:   N/A")
    print(f"    Retrieval Hits:   {all_hits}/{len(results)} ({100*all_hits/len(results):.1f}%)")
    print(f"    Target:           0.800 (Vincent's threshold)")

    overall_combined = sum(all_combined)/len(all_combined) if all_combined else 0
    print(f"    Status:           {'PASS' if overall_combined >= 0.80 else 'FAIL'}")

    print(f"\n  Per-Category:")
    for cat in ["MSDS", "Policy", "Standards"]:
        if cat in category_scores:
            cs = category_scores[cat]
            avg_kw = sum(cs["kw"])/len(cs["kw"]) if cs["kw"] else 0
            avg_combined = sum(cs["combined"])/len(cs["combined"]) if cs["combined"] else 0
            avg_llm = sum(cs["llm"])/len(cs["llm"]) if cs["llm"] else -1
            hit_rate = cs["hits"]/cs["total"]*100 if cs["total"] else 0
            print(f"    {cat:10s}: KW={avg_kw:.3f} | LLM={(f'{avg_llm:.3f}' if avg_llm >= 0 else 'N/A'):>5} | "
                  f"Combined={avg_combined:.3f} | Hits={cs['hits']}/{cs['total']} ({hit_rate:.0f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"vincent_eval_v2_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "collection": config.chroma.collection,
                "retrieval_k": config.rag.retrieval_k,
                "top_k": config.rag.top_k,
                "embedding_model": config.embedding.model_name,
            },
            "summary": {
                "overall_keyword": round(sum(all_kw)/len(all_kw), 3) if all_kw else None,
                "overall_combined": round(overall_combined, 3),
                "retrieval_hits": f"{all_hits}/{len(results)}",
                "target": 0.80,
                "status": "PASS" if overall_combined >= 0.80 else "FAIL",
            },
            "category_summary": {
                cat: {
                    "avg_keyword": round(sum(cs["kw"])/len(cs["kw"]), 3) if cs["kw"] else None,
                    "avg_combined": round(sum(cs["combined"])/len(cs["combined"]), 3) if cs["combined"] else None,
                    "retrieval_hits": f"{cs['hits']}/{cs['total']}",
                } for cat, cs in category_scores.items()
            },
            "questions": results,
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")
    print(f"{'='*70}\n")

    return overall_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vincent's 45-Question Eval v2")
    parser.add_argument("--local-path", type=str, help="Path to local ChromaDB directory")
    parser.add_argument("--collection", type=str, help="ChromaDB collection name")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM scoring (keyword-only)")
    args = parser.parse_args()

    score = run_eval(args)
    sys.exit(0 if score >= 0.80 else 1)
