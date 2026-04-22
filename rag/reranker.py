"""
Domain-aware reranker for O'Connors IMS document retrieval.

After ChromaDB returns the initial top-k results by vector similarity,
this reranker adjusts scores based on:
    1. Subfolder relevance — queries about procedures should prefer
       chunks from the Procedures folder over Australian Standards.
    2. Filename keyword matching — if the query mentions "SOP-064",
       chunks from files containing "SOP-064" get a boost.
    3. Source diversity — prevents all top results coming from one
       massive document (e.g., a 300-page Australian Standard).

This addresses the core faithfulness problem: at 44k+ chunks, Australian
Standards dominate retrieval by volume, starving Policies/SOPs/SWMSs
of representation in the top results.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A retrieval result after reranking."""
    chunk_id: str
    document: str
    metadata: dict
    original_distance: float
    reranked_score: float
    boost_reasons: list


class Reranker:
    """
    Domain-aware reranker for O'Connors IMS retrieval results.

    Applies three scoring adjustments on top of ChromaDB's cosine distance:
        1. Subfolder hints — maps query intent to preferred subfolders.
        2. Filename keyword boost — exact matches on document codes.
        3. Source diversity — limits results from any single file.
    """

    # Maps query keywords to preferred subfolders with boost weights.
    # Higher weight = stronger preference for that subfolder.
    SUBFOLDER_HINTS = {
        "procedure": {"Procedures": 0.15, "SOP's": 0.10},
        "policy": {"Policies": 0.20},
        "sop": {"SOP's": 0.20},
        "swms": {"SWMS's": 0.20},
        "safe work": {"SWMS's": 0.15, "SOP's": 0.10},
        "form": {"Forms": 0.15},
        "standard": {"Australian Standards": 0.10},
        "msds": {"MSDS's": 0.20},
        "safety data": {"MSDS's": 0.15},
        "emergency": {"Procedures": 0.15},
        "first aid": {"Procedures": 0.15},
        "ppe": {"SOP's": 0.15, "SWMS's": 0.10},
        "hazard": {"SOP's": 0.10, "SWMS's": 0.10, "MSDS's": 0.10},
        "refrigerant": {"SOP's": 0.15},
        "commissioning": {"Procedures": 0.15},
        "maintenance": {"SOP's": 0.15},
        "call out": {"Procedures": 0.20},
        "after hours": {"Procedures": 0.20},
        "incident": {"Procedures": 0.15, "Forms": 0.10},
        "induction": {"SOP's": 0.15, "Forms": 0.10},
        "isolation": {"Procedures": 0.15, "SOP's": 0.10},
        "lock out": {"Procedures": 0.15},
        "working at height": {"SWMS's": 0.15, "Procedures": 0.10},
        "confined space": {"SWMS's": 0.15, "SOP's": 0.10},
        "hot work": {"SWMS's": 0.15},
        "permit": {"Forms": 0.10, "SOP's": 0.10},
        "inspection": {"SOP's": 0.10, "Forms": 0.10},
        "test": {"Procedures": 0.10},
        "chiller": {"SOP's": 0.15},
        "hvac": {"SOP's": 0.10, "Australian Standards": 0.05},
        "fire": {"Australian Standards": 0.10, "SOP's": 0.10},
        "electrical": {"Australian Standards": 0.10, "Forms": 0.05},
    }

    # Document code patterns to detect in queries (e.g., "SOP-064", "CP01", "WP03")
    DOC_CODE_PATTERN = re.compile(
        r'\b(SOP[-\s]?\d+|CP\d+|SP\d+|WP\d+|CTP[-\s]?\d+|EP\d+|'
        r'AHP\d+|FP\d+|DP\d+|RP\d+|PSP\d+|RWP\d+|FCP\d+|'
        r'SWMS[-\s]?\d+|COM\.?P\s?\d+|S_SP\s?\d+|HSE[-\s]?\d+|'
        r'AS\s?\d+[\.\d]*)',
        re.IGNORECASE
    )

    def __init__(
        self,
        subfolder_boost_weight: float = 1.0,
        keyword_boost_weight: float = 1.5,
        max_per_source: int = 3,
    ):
        self._subfolder_weight = subfolder_boost_weight
        self._keyword_weight = keyword_boost_weight
        self._max_per_source = max_per_source

    def _compute_subfolder_boost(self, query: str, subfolder: str) -> tuple[float, str]:
        """Check if query keywords suggest a preferred subfolder."""
        query_lower = query.lower()
        total_boost = 0.0
        reasons = []

        for keyword, folder_weights in self.SUBFOLDER_HINTS.items():
            if keyword in query_lower and subfolder in folder_weights:
                boost = folder_weights[subfolder] * self._subfolder_weight
                total_boost += boost
                reasons.append(f"subfolder '{keyword}'→{subfolder} +{boost:.2f}")

        return total_boost, reasons

    def _compute_keyword_boost(self, query: str, source_file: str) -> tuple[float, str]:
        """Boost chunks from files whose name matches document codes in the query."""
        codes = self.DOC_CODE_PATTERN.findall(query)
        if not codes:
            return 0.0, []

        source_upper = source_file.upper().replace(" ", "").replace("-", "")
        total_boost = 0.0
        reasons = []

        for code in codes:
            code_clean = code.upper().replace(" ", "").replace("-", "")
            if code_clean in source_upper:
                boost = 0.20 * self._keyword_weight
                total_boost += boost
                reasons.append(f"code match '{code}' in {source_file[:40]} +{boost:.2f}")

        return total_boost, reasons

    def rerank(
        self,
        query: str,
        results: list[dict],
        final_top_k: int = 5,
    ) -> list[RankedResult]:
        """
        Rerank retrieval results using domain-aware scoring.

        Args:
            query: The user's original question.
            results: List of dicts with keys: chunk_id, document, metadata, distance.
            final_top_k: Number of results to return after reranking.

        Returns:
            Reranked list of RankedResult objects, sorted by score (highest first).
        """
        scored = []

        for r in results:
            metadata = r.get("metadata", {})
            subfolder = metadata.get("subfolder", "")
            source_file = metadata.get("source_file", "")
            distance = r.get("distance", 1.0)

            # Base score: invert distance (lower distance = higher score)
            # Cosine distance range is [0, 2], typical results are [0, 1]
            base_score = max(0, 1.0 - distance)

            boosts = []

            # Subfolder boost
            sf_boost, sf_reasons = self._compute_subfolder_boost(query, subfolder)
            base_score += sf_boost
            boosts.extend(sf_reasons)

            # Keyword boost
            kw_boost, kw_reasons = self._compute_keyword_boost(query, source_file)
            base_score += kw_boost
            boosts.extend(kw_reasons)

            scored.append(RankedResult(
                chunk_id=r.get("chunk_id", ""),
                document=r.get("document", ""),
                metadata=metadata,
                original_distance=distance,
                reranked_score=base_score,
                boost_reasons=boosts,
            ))

        # Sort by reranked score (highest first)
        scored.sort(key=lambda x: x.reranked_score, reverse=True)

        # Apply source diversity: limit results from any single file
        if self._max_per_source > 0:
            source_counts = {}
            diverse = []
            overflow = []

            for item in scored:
                src = item.metadata.get("source_file", "unknown")
                count = source_counts.get(src, 0)
                if count < self._max_per_source:
                    diverse.append(item)
                    source_counts[src] = count + 1
                else:
                    overflow.append(item)

            # Fill remaining slots from overflow if needed
            scored = diverse + overflow

        final = scored[:final_top_k]

        if final:
            logger.info(
                "Reranked: top score=%.3f (%s), boosts=%s",
                final[0].reranked_score,
                final[0].metadata.get("source_file", "?")[:40],
                final[0].boost_reasons or "none",
            )

        return final
