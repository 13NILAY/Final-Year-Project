"""
Canonical ESG Metrics and Alias Management
===========================================
Loads the refined metric schema and provides methods to map surface phrases
to canonical metric names using exact, fuzzy, and embedding-based similarity.
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

# Optional: sentence-transformers for embedding similarity
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("  [AliasManager] sentence-transformers not installed; embedding similarity disabled.")


class AliasManager:
    """Loads and manages canonical metric definitions and aliases."""

    def __init__(self, json_path: Optional[str] = None, use_embeddings: bool = True):
        if json_path is None:
            json_path = os.path.join(os.path.dirname(__file__), "canonical_metrics.json")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.metrics = data["metrics"]

        # Build alias lookup (lowercased)
        self.alias_to_canonical = {}
        self.all_phrases = []          # list of (phrase, canonical) for embedding
        for canonical, info in self.metrics.items():
            # Include the canonical name itself as a phrase
            canonical_phrase = canonical.replace('_', ' ')
            self.alias_to_canonical[canonical_phrase.lower()] = canonical
            self.all_phrases.append((canonical_phrase, canonical))
            for alias in info.get("aliases", []):
                normalized = re.sub(r'\s+', ' ', alias.lower().strip())
                self.alias_to_canonical[normalized] = canonical
                self.all_phrases.append((alias, canonical))

        self.canonical_names = set(self.metrics.keys())

        # Embedding model (lazy loaded)
        self._model = None
        self._phrase_embeddings = None
        self.use_embeddings = use_embeddings and EMBEDDING_AVAILABLE
        if self.use_embeddings:
            self._load_embeddings()

    def _load_embeddings(self):
        """Load sentence transformer and precompute phrase embeddings."""
        print("  [AliasManager] Loading embedding model and precomputing...")
        self._model = SentenceTransformer('all-MiniLM-L6-v2')
        # Precompute embeddings for all phrases
        phrases = [p for p, _ in self.all_phrases]
        self._phrase_embeddings = self._model.encode(phrases, convert_to_tensor=True, show_progress_bar=False)
        print(f"  [AliasManager] Precomputed embeddings for {len(phrases)} phrases.")

    def get_canonical(self, phrase: str, threshold: float = 0.85) -> Optional[str]:
        """
        Map a surface phrase to a canonical metric name.
        First tries exact match (after normalization), then fuzzy matching,
        then embedding similarity if enabled.
        """
        if not phrase:
            return None

        # Normalize input
        normalized = re.sub(r'\s+', ' ', phrase.lower().strip())

        # 1. Exact match
        if normalized in self.alias_to_canonical:
            return self.alias_to_canonical[normalized]

        # 2. Fuzzy match
        best_candidate = None
        best_score = 0.0
        for alias, canonical in self.alias_to_canonical.items():
            score = SequenceMatcher(None, normalized, alias).ratio()
            if score > best_score:
                best_score = score
                best_candidate = canonical
        if best_score >= threshold:
            return best_candidate

        # 3. Embedding similarity (if enabled)
        if self.use_embeddings:
            similar = self.get_similar(phrase, top_k=1, threshold=threshold)
            if similar:
                return similar[0][0]  # (canonical, score)

        return None

    def get_similar(self, text: str, top_k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find top_k most similar canonical metrics by embedding similarity.
        Returns list of (canonical, score).
        """
        if not self.use_embeddings or self._model is None:
            return []

        # Encode input text
        emb = self._model.encode([text], convert_to_tensor=True)

        # Compute cosine similarities with all phrase embeddings
        similarities = (emb @ self._phrase_embeddings.T).squeeze(0)

        # Get top_k indices
        top_scores, top_indices = similarities.topk(min(top_k, len(self._phrase_embeddings)))

        # Collect results, deduplicate canonical metrics
        seen = set()
        results = []
        for score, idx in zip(top_scores, top_indices):
            canonical = self.all_phrases[idx][1]
            if canonical not in seen and score.item() >= threshold:
                seen.add(canonical)
                results.append((canonical, score.item()))
        return results

    def get_all_metrics(self) -> Dict[str, Dict]:
        return self.metrics

    def get_by_category(self, category: str) -> Dict[str, Dict]:
        return {k: v for k, v in self.metrics.items() if v["category"] == category}

    def get_expected_unit(self, canonical: str) -> Optional[str]:
        return self.metrics.get(canonical, {}).get("expected_unit")

    def get_allowed_qualifiers(self, canonical: str) -> List[str]:
        return self.metrics.get(canonical, {}).get("allowed_qualifiers", [])


# Singleton instance with optional embedding
_ALIAS_MANAGER = None

def get_alias_manager(use_embeddings: bool = True) -> AliasManager:
    """Singleton accessor for the alias manager."""
    global _ALIAS_MANAGER
    if _ALIAS_MANAGER is None:
        _ALIAS_MANAGER = AliasManager(use_embeddings=use_embeddings)
    return _ALIAS_MANAGER