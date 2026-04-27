"""
Answer generator using HuggingFace Inference API.

Takes retrieved and reranked document chunks, builds a domain-specific
prompt, and generates a grounded answer via HuggingFace InferenceClient.
Includes confidence scoring based on retrieval quality.

CHANGES (v2.1 — 24 Apr 2026):
  - Default temperature changed from 0.1 to 0.0 for consistent answers
  - Sources output now includes clean document names (not raw paths)
  - Removed chunk_id from client-facing sources (internal detail)
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import InferenceClient

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
    Generates answers from retrieved IMS document chunks using HuggingFace Inference API.

    The generator:
        1. Formats chunks into a structured context with source labels.
        2. Builds a system + user prompt tuned for faithfulness.
        3. Calls the HuggingFace Inference API to generate an answer.
        4. Computes a confidence score based on retrieval distances.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
        if not token:
            raise RuntimeError("HUGGING_FACE_HUB_TOKEN environment variable is not set")

        self._client = InferenceClient(model=self._model, token=token)
        logger.info("HuggingFace InferenceClient initialized with model: %s", self._model)

    def _compute_confidence(self, reranked_results: list) -> float:
        if not reranked_results:
            return 0.0

        scores = [r.reranked_score for r in reranked_results]
        avg_score = sum(scores) / len(scores)
        confidence = min(1.0, max(0.0, avg_score))

        if reranked_results[0].boost_reasons:
            confidence = min(1.0, confidence + 0.05)

        return round(confidence, 3)

    @staticmethod
    def _clean_source_name(raw_path: str, subfolder: str = "") -> str:
        """Convert raw file path to clean document name for API response."""
        name = os.path.basename(raw_path)
        name = os.path.splitext(name)[0]
        name = re.sub(r'\s*ID\s+\d{2}[\s.]\d{2}[\s.]\d{2,4}', '', name)
        name = re.sub(r'\s*_?\s*v\.?\d+\s+\d{2}\.\d{2}\.\d{2,4}', '', name)
        name = re.sub(r'\s*_?\s*Issue\s+\d+\s+\d{2}\s+\d{2}\s+\d{4}', '', name)
        name = re.sub(r'\s*_?\s*REV\s+\w+\s*', ' ', name)
        name = name.replace(' _ ', ' — ').replace('_', ' ')
        name = re.sub(r'\s+', ' ', name).strip()
        if subfolder and "MSDS" in subfolder and "MSDS" not in name:
            name = f"{name} MSDS"
        return name

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
        chunks = []
        sources = []
        seen_files = set()

        for r in reranked_results:
            raw_file = r.metadata.get("source_file", "Unknown")
            subfolder = r.metadata.get("subfolder", "")

            chunks.append({
                "text": r.document,
                "source_file": raw_file,
                "subfolder": subfolder,
            })

            # Deduplicate sources — only list each document once
            if raw_file not in seen_files:
                seen_files.add(raw_file)
                sources.append({
                    "document": self._clean_source_name(raw_file, subfolder),
                    "category": subfolder,
                })

        # Build prompt
        system_prompt, user_prompt = PromptTemplates.build_prompt(question, chunks)
        confidence = self._compute_confidence(reranked_results)

        try:
            response = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            answer = response.choices[0].message.content

            return GeneratedAnswer(
                answer=answer,
                sources=sources,
                confidence=confidence,
                model=self._model,
                prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                eval_tokens=getattr(response.usage, "completion_tokens", None),
            )

        except Exception as e:
            logger.error("HuggingFace generation failed: %s", e)
            return GeneratedAnswer(
                answer=(
                    "An error occurred while processing your question. "
                    "Please try again or contact your supervisor."
                ),
                sources=sources,
                confidence=confidence,
                model=self._model,
            )
