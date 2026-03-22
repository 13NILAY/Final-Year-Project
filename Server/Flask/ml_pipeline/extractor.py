"""
ML Extractor Module (Staged Pipeline)
=======================================
Hybrid extraction using sentences/windows, value extraction, qualifiers,
canonical mapping, and conflict resolution.
"""

import re
import torch
from typing import Dict, List, Optional, Tuple
from .canonical_metrics import get_alias_manager
from .qualifiers import extract_qualifiers
from .labeling import ESG_METRICS  # still used for value extraction patterns
from .preprocessing import clean_text, chunk_text


# Custom sentence tokenizer (replaces NLTK)
def _simple_sent_tokenize(text: str) -> List[str]:
    """
    Split text into sentences using punctuation and capital letter detection.
    Handles common abbreviations and newlines.
    """
    if not text:
        return []
    # Split on .!? followed by whitespace and a capital letter
    # Also splits on newlines
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)', text)
    # Clean up
    result = [s.strip() for s in sentences if s.strip()]
    # If the result is empty, just return the original text as one sentence
    if not result:
        return [text.strip()]
    return result


class ESGStagedExtractor:
    """
    Staged ESG metric extraction pipeline.
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.25):
        self.alias_manager = get_alias_manager()
        self.confidence_threshold = confidence_threshold
        # For candidate detection: lightweight ESG keyword set
        self.esg_keywords = {
            'emissions', 'ghg', 'scope', 'energy', 'renewable', 'water', 'waste',
            'recycled', 'hazardous', 'turnover', 'attrition', 'female', 'women',
            'training', 'safety', 'injury', 'satisfaction', 'csr', 'community',
            'board', 'director', 'independence', 'pay ratio', 'ethics', 'whistleblower'
        }

        # Load ML model if available (for future use)
        self.ml_model = None
        if model_path:
            # Placeholder for future ML integration (e.g., a fasttext classifier)
            pass

    def extract_from_text(self, text: str, page_map: Optional[Dict[int, str]] = None) -> Dict:
        """
        Main extraction method.
        Args:
            text: Cleaned text from a PDF.
            page_map: Optional mapping from sentence indices to page numbers.
        Returns:
            Dictionary of extracted metrics, keyed by canonical metric name.
        """
        # 1. Split into sentences and create windows
        sentences = _simple_sent_tokenize(text)
        windows = self._create_windows(sentences, window_size=3)

        all_candidates = []

        for window in windows:
            # 2. Candidate detection: must have a number and an ESG keyword
            if not self._is_esg_candidate(window):
                continue

            # 3. Value & unit extraction (using patterns from labeling)
            values = self._extract_values_from_window(window)
            if not values:
                continue

            # 4. For each value, extract qualifiers
            for val_info in values:
                qualifiers = extract_qualifiers(window)
                # Add window context and page number
                candidate = {
                    'text': window,
                    'value': val_info['value'],
                    'unit': val_info['unit'],
                    'raw_match': val_info['raw_match'],
                    'qualifiers': qualifiers,
                    'page': self._guess_page(window, sentences, page_map),
                    'confidence': 0.5  # placeholder, will be refined
                }
                all_candidates.append(candidate)

        # 5. Canonical mapping and conflict resolution
        extracted = self._map_and_resolve(all_candidates)

        return extracted

    def _create_windows(self, sentences: List[str], window_size: int = 3) -> List[str]:
        """Create overlapping windows of sentences."""
        windows = []
        for i in range(0, len(sentences), window_size // 2):
            window = ' '.join(sentences[i:i+window_size])
            if window.strip():
                windows.append(window)
        return windows

    def _is_esg_candidate(self, text: str) -> bool:
        """Check if text contains a number and at least one ESG keyword."""
        if not re.search(r'\d', text):
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.esg_keywords)

    def _extract_values_from_window(self, text: str) -> List[Dict]:
        """
        Extract all numeric values with their units from the window.
        Uses regex patterns from ESG_METRICS (old labeling) but does not assign metrics yet.
        Returns list of dicts with 'value', 'unit', 'raw_match'.
        """
        results = []
        # For simplicity, we iterate over all metric patterns and collect matches.
        # In a production system, we might use a unified number+unit extraction.
        for metric_name, metric_def in ESG_METRICS.items():
            for pattern in metric_def['patterns']:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                except re.error:
                    continue
                for match in matches:
                    try:
                        value_str = match.group(1).replace(',', '').strip()
                        # Handle ratio like 85:1
                        if ':' in value_str:
                            value_str = value_str.split(':')[0].strip()
                        value = float(value_str)
                        # Apply multipliers (simplified)
                        # (We'll keep the logic from original extract_value_and_unit)
                        # For brevity, we'll call a helper
                        value = self._apply_multipliers(value, match.group(0), text)
                        results.append({
                            'value': value,
                            'unit': metric_def['unit'],
                            'raw_match': match.group(0)
                        })
                    except (ValueError, IndexError):
                        continue
        return results

    def _apply_multipliers(self, value: float, match_text: str, context: str) -> float:
        """Apply thousand/million/billion multipliers based on surrounding text."""
        combined = (match_text + ' ' + context).lower()
        if 'million' in combined and value < 1000:
            value *= 1_000_000
        elif 'billion' in combined and value < 1000:
            value *= 1_000_000_000
        elif 'thousand' in combined and value < 10000:
            value *= 1_000
        elif 'lakh' in combined and value < 100000:
            value *= 100_000
        elif 'crore' in combined and value < 100000:
            value *= 10_000_000
        return value

    def _guess_page(self, window: str, sentences: List[str], page_map: Optional[Dict]) -> Optional[int]:
        """Simple heuristic: find first sentence of window in page_map."""
        if not page_map:
            return None
        first_sent = window.split('.')[0].strip()
        for idx, sent in enumerate(sentences):
            if sent.startswith(first_sent) or first_sent in sent:
                return page_map.get(idx)
        return None

    def _map_and_resolve(self, candidates: List[Dict]) -> Dict:
        """
        Map candidates to canonical metrics, apply validation rules,
        and deduplicate.
        """
        extracted = {}
        for cand in candidates:
            # For now, use alias manager on the raw match phrase
            # (In future, we could also use the full window for context)
            raw_phrase = cand['raw_match']
            canonical = self.alias_manager.get_canonical(raw_phrase)
            if not canonical:
                # Try to extract a noun phrase from window?
                # Fallback: skip
                continue

            # Apply validation rules (example: water recycled should not map to withdrawal)
            if not self._validate_mapping(canonical, cand):
                continue

            # Compute confidence (placeholder)
            confidence = self._compute_confidence(cand, canonical)

            # Store, keeping highest confidence
            if canonical not in extracted or confidence > extracted[canonical]['confidence']:
                extracted[canonical] = {
                    'canonical_metric': canonical,
                    'value': cand['value'],
                    'raw_unit': cand['unit'],
                    'normalized_unit': self._normalize_unit(cand['unit'], canonical),
                    'year': cand['qualifiers']['year'],
                    'scope': cand['qualifiers']['scope'],
                    'actual_or_target': cand['qualifiers']['actual_or_target'],
                    'geography': cand['qualifiers']['geography'],
                    'entity_level': cand['qualifiers']['entity_level'],
                    'page': cand['page'],
                    'source_text': cand['text'],
                    'confidence': confidence,
                    'extraction_method': 'regex+alias',
                }
        return extracted

    def _validate_mapping(self, canonical: str, candidate: Dict) -> bool:
        """Rule-based validation to reject impossible mappings."""
        text = candidate['text'].lower()
        # Rule: if phrase contains 'recycled' but canonical is water withdrawal, reject
        if 'recycled' in text and canonical in ['water_withdrawal_total', 'water_discharge_total']:
            return False
        # Rule: if phrase contains 'board' or 'director' but canonical is not governance, reject
        if ('board' in text or 'director' in text) and self.alias_manager.metrics[canonical]['category'] != 'governance':
            return False
        # Rule: if phrase contains 'scope' but canonical does not have scope qualifier, still ok, but we keep
        # (additional rules can be added)
        return True

    def _compute_confidence(self, candidate: Dict, canonical: str) -> float:
        """Compute confidence score (0-1)."""
        # Simple heuristic: if value extracted and alias matched, base confidence 0.7
        conf = 0.7
        # Bonus if qualifiers are present
        if candidate['qualifiers']['year']:
            conf += 0.1
        if candidate['qualifiers']['scope'] is not None:
            conf += 0.1
        # Cap at 1.0
        return min(conf, 1.0)

    def _normalize_unit(self, raw_unit: str, canonical: str) -> str:
        """Normalize unit to expected unit for the canonical metric."""
        # Simple mapping
        mapping = {
            'tco2e': 'tCO2e',
            't co2e': 'tCO2e',
            'tonnes co2': 'tCO2e',
            'kl': 'm3',
            'kilolitres': 'm3',
            'm3': 'm3',
            'mwh': 'MWh',
            'gwh': 'MWh',
            'kwh': 'MWh',
            'gj': 'MWh',  # approximate, but we keep as is for now
        }
        norm = mapping.get(raw_unit.lower().strip(), raw_unit)
        # If canonical expects a specific unit, we might convert, but keep it simple
        expected = self.alias_manager.get_expected_unit(canonical)
        if expected and norm != expected:
            # Optionally try to convert (e.g., tonnes to tCO2e if both mass units)
            pass
        return norm