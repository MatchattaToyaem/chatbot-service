"""
Answer generator using Ollama llama3.2.

Takes retrieved and reranked document chunks, builds a domain-specific
prompt, and generates a grounded answer via the local Ollama API.
Includes confidence scoring based on retrieval quality.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from rag.prompts import PromptTemplates

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """A generated answer with source attribution and confidence."""
    answer: str
    sources: list[dict]
    confidence: float
    model: str
    prompt_tokens: Optional[int] = None
    eval_tokens: Optional[int] = None


class AnswerGenerator:
    """
    Generates answers from retrieved IMS document chunks using Ollama.

    The generator:
        1. Formats chunks into a structured context with source labels.
        2. Builds a system + user prompt tuned for faithfulness.
        3. Calls the local Ollama API to generate an answer.
        4. Computes a confidence score based on retrieval distances.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _check_ollama(self) -> bool:
        """Verify Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(self._model in m for m in models):
                    return True
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    self._model, models,
                )
            return False
        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama at %s", self._base_url)
            return False

    def _compute_confidence(self, reranked_results: list) -> float:
        """
        Compute a confidence score based on retrieval quality.

        Factors:
            - Average reranked score of top results (higher = more relevant)
            - Score gap between #1 and #2 (larger gap = more confident)
            - Number of results with boosts (domain relevance)
        """
        if not reranked_results:
            return 0.0

        scores = [r.reranked_score for r in reranked_results]
        avg_score = sum(scores) / len(scores)

        # Normalise to [0, 1] range — typical scores are 0.3 to 1.2
        confidence = min(1.0, max(0.0, avg_score))

        # Boost confidence if top result has domain-specific boosts
        if reranked_results[0].boost_reasons:
            confidence = min(1.0, confidence + 0.05)

        return round(confidence, 3)

    def generate(
        self,
        question: str,
        reranked_results: list,
    ) -> GeneratedAnswer:
        """
        Generate an answer from reranked retrieval results.

        Args:
            question: The user's question.
            reranked_results: List of RankedResult objects from the reranker.

        Returns:
            GeneratedAnswer with the response, sources, and confidence.
        """
        # Prepare chunks for prompt
        chunks = []
        sources = []
        for r in reranked_results:
            chunks.append({
                "text": r.document,
                "source_file": r.metadata.get("source_file", "Unknown"),
                "subfolder": r.metadata.get("subfolder", ""),
            })
            sources.append({
                "file": r.metadata.get("source_file", "Unknown"),
                "subfolder": r.metadata.get("subfolder", ""),
                "score": round(r.reranked_score, 3),
                "chunk_id": r.chunk_id,
            })

        # Build prompt
        system_prompt, user_prompt = PromptTemplates.build_prompt(question, chunks)

        # Compute confidence before generation (based on retrieval quality)
        confidence = self._compute_confidence(reranked_results)

        # Check Ollama availability
        if not self._check_ollama():
            return GeneratedAnswer(
                answer=(
                    "Ollama is not available. Please ensure Ollama is running "
                    f"with the {self._model} model loaded.\n\n"
                    "Retrieved context is available — the retrieval pipeline "
                    "is working, but answer generation requires a local LLM."
                ),
                sources=sources,
                confidence=confidence,
                model=self._model,
            )

        # Call Ollama API
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("message", {}).get("content", "No response generated.")

            return GeneratedAnswer(
                answer=answer,
                sources=sources,
                confidence=confidence,
                model=self._model,
                prompt_tokens=data.get("prompt_eval_count"),
                eval_tokens=data.get("eval_count"),
            )

        except requests.Timeout:
            logger.error("Ollama request timed out after 120s")
            return GeneratedAnswer(
                answer="The LLM request timed out. Try a simpler question or check Ollama status.",
                sources=sources,
                confidence=confidence,
                model=self._model,
            )
        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return GeneratedAnswer(
                answer=f"Generation failed: {e}",
                sources=sources,
                confidence=confidence,
                model=self._model,
            )
