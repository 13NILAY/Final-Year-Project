"""
ML Extractor Module
===================
RoBERTa-based ESG metric classifier with regex value extraction.
Hybrid approach: ML for metric identification + regex for value/unit extraction.
Multi-head architecture with Environmental/Social/Governance classifier heads.
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import RobertaTokenizer, RobertaModel

from .labeling import (
    ESG_METRICS, METRIC_NAMES, METRIC_TO_ID, ID_TO_METRIC, NUM_CLASSES, CATEGORY_MAP
)
from .preprocessing import chunk_text, clean_text


# ─── METRIC TYPE SETS (for value extraction safety) ─────────────────────────

# Metrics that represent ratios — should NEVER be scaled by magnitude multipliers
RATIO_METRICS = {'ceo_pay_ratio', 'lost_time_injury'}

# Metrics that represent percentages — should NEVER be scaled
PERCENTAGE_METRICS = {
    'renewable_energy', 'waste_recycled', 'employee_turnover',
    'female_representation', 'board_independence', 'female_directors',
    'ethics_training', 'employee_satisfaction',
}


# Sanity bounds per metric type for post-processing normalization
METRIC_VALUE_BOUNDS = {
    'ceo_pay_ratio': (1, 500),
    'lost_time_injury': (0, 50),
    'renewable_energy': (0, 100),
    'waste_recycled': (0, 100),
    'employee_turnover': (0, 100),
    'female_representation': (0, 100),
    'board_independence': (0, 100),
    'female_directors': (0, 100),
    'ethics_training': (0, 100),
    'employee_satisfaction': (0, 100),
    'ghg_emissions': (0, 1e12),
    'scope1_emissions': (0, 1e12),
    'scope2_emissions': (0, 1e12),
    'scope3_emissions': (0, 1e12),
    'co2_emissions': (0, 1e12),
    'energy_consumption': (0, 1e12),
    'water_withdrawal': (0, 1e12),
    'hazardous_waste': (0, 1e12),
    'training_hours': (0, 1000),
    'whistleblower_cases': (0, 100000),
    'community_investment': (0, 1e12),
}


# ─── MODEL DEFINITION ───────────────────────────────────────────────────────

class ESGMetricClassifier(nn.Module):
    """
    RoBERTa-based classifier for ESG metric identification.
    
    Multi-head architecture:
        RoBERTa encoder → [CLS] pooling → shared dropout →
            → Environmental head → env metrics
            → Social head → social metrics
            → Governance head → governance metrics
            → Combined logits → NUM_CLASSES
    
    Predicts which ESG metric (or 'no_metric') a text chunk describes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        hidden_size = self.bert.config.hidden_size  # 768 for roberta-base
        self.dropout = nn.Dropout(dropout)
        
        # Category-specific metric indices
        env_metrics = CATEGORY_MAP.get('environmental', [])
        soc_metrics = CATEGORY_MAP.get('social', [])
        gov_metrics = CATEGORY_MAP.get('governance', [])
        
        self.env_indices = [METRIC_TO_ID[m] for m in env_metrics if m in METRIC_TO_ID]
        self.soc_indices = [METRIC_TO_ID[m] for m in soc_metrics if m in METRIC_TO_ID]
        self.gov_indices = [METRIC_TO_ID[m] for m in gov_metrics if m in METRIC_TO_ID]
        self.no_metric_idx = METRIC_TO_ID.get('no_metric', num_classes - 1)
        
        # Multi-head classifiers
        # Environmental head: hidden_size → len(env_metrics)
        self.env_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(self.env_indices)),
        )
        
        # Social head: hidden_size → len(soc_metrics)
        self.soc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(self.soc_indices)),
        )
        
        # Governance head: hidden_size → len(gov_metrics)
        self.gov_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, len(self.gov_indices)),
        )
        
        # No-metric head (single output for 'no_metric' class)
        self.no_metric_head = nn.Linear(hidden_size, 1)
        
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token) representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Get logits from each head
        env_logits = self.env_head(cls_output)   # (batch, len(env_indices))
        soc_logits = self.soc_head(cls_output)   # (batch, len(soc_indices))
        gov_logits = self.gov_head(cls_output)   # (batch, len(gov_indices))
        no_metric_logit = self.no_metric_head(cls_output)  # (batch, 1)
        
        # Assemble full logits tensor
        batch_size = input_ids.size(0)
        full_logits = torch.zeros(batch_size, self.num_classes, device=input_ids.device)
        
        for i, idx in enumerate(self.env_indices):
            full_logits[:, idx] = env_logits[:, i]
        for i, idx in enumerate(self.soc_indices):
            full_logits[:, idx] = soc_logits[:, i]
        for i, idx in enumerate(self.gov_indices):
            full_logits[:, idx] = gov_logits[:, i]
        full_logits[:, self.no_metric_idx] = no_metric_logit.squeeze(-1)
        
        return full_logits

    def predict(self, input_ids, attention_mask) -> Tuple[List[str], List[float]]:
        """
        Predict metric names with confidence scores.
        
        Returns:
            (predicted_metric_names, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probas = torch.softmax(logits, dim=-1)
            confidences, predicted_ids = torch.max(probas, dim=-1)

        metric_names = [ID_TO_METRIC[idx.item()] for idx in predicted_ids]
        conf_scores = confidences.cpu().numpy().tolist()
        return metric_names, conf_scores


# ─── VALUE & UNIT EXTRACTION (REGEX) ─────────────────────────────────────────

def _extract_scope_number(text: str, start: int, end: int) -> Optional[int]:
    """
    Extract the scope number (1, 2, or 3) from the nearby context of a match.
    Uses a narrow ±30 character window to prevent scope cross-contamination.
    """
    window_start = max(0, start - 30)
    window_end = min(len(text), end + 30)
    context = text[window_start:window_end].lower()
    
    # Look for explicit scope number
    scope_match = re.search(r'scope\s*([123])', context)
    if scope_match:
        return int(scope_match.group(1))
    
    # Look for word-form scope
    if 'scope one' in context or 'direct emission' in context:
        return 1
    if 'scope two' in context or 'indirect emission' in context:
        return 2
    if 'scope three' in context or 'value chain' in context or 'other indirect' in context:
        return 3
    
    return None


def normalize_metric_value(value: float, metric_name: str) -> Optional[float]:
    """
    Post-processing normalization: ensure extracted values are within
    reasonable bounds for each metric type.
    
    Returns:
        Normalized value, or None if value is clearly invalid
    """
    if metric_name in METRIC_VALUE_BOUNDS:
        min_val, max_val = METRIC_VALUE_BOUNDS[metric_name]
        if value < min_val or value > max_val:
            return None  # Value is out of range — likely extraction error
    return value


def extract_value_and_unit(text: str, metric_name: str) -> Optional[Dict]:
    """
    Extract numeric value and unit from text for a given metric.
    Uses metric-specific regex patterns from the labeling module.
    
    Includes fixes for:
    - CEO pay ratio magnitude bug (no million/billion scaling for ratios)
    - Scope disambiguation (narrow context window)
    - Percentage metrics clamping
    
    Args:
        text: Text chunk to extract from
        metric_name: The identified ESG metric name
        
    Returns:
        Dict with 'value', 'unit', 'raw_match' or None
    """
    if metric_name not in ESG_METRICS:
        return None

    metric_def = ESG_METRICS[metric_name]
    text_lower = text.lower()

    for pattern in metric_def['patterns']:
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE))
        for match in matches:
            try:
                value_str = match.group(1).replace(',', '').strip()
                
                # Handle ratio format like "85:1" — extract number before ':'
                if ':' in value_str:
                    value_str = value_str.split(':')[0].strip()
                
                value = float(value_str)

                # Scope disambiguation: verify the scope number matches
                if metric_name in ('scope1_emissions', 'scope2_emissions', 'scope3_emissions'):
                    expected_scope = int(metric_name.replace('scope', '').replace('_emissions', ''))
                    detected_scope = _extract_scope_number(text, match.start(), match.end())
                    if detected_scope is not None and detected_scope != expected_scope:
                        continue  # Wrong scope — skip this match

                # --- FIX: Do NOT apply multipliers for ratio/percentage metrics ---
                if metric_name not in RATIO_METRICS and metric_name not in PERCENTAGE_METRICS:
                    match_text = match.group(0).lower()
                    context_window = text_lower[max(0, match.start()-50):match.end()+50]

                    if 'million' in match_text or 'million' in context_window:
                        if value < 1000:
                            value *= 1_000_000
                    elif 'billion' in match_text or 'billion' in context_window:
                        if value < 1000:
                            value *= 1_000_000_000
                    elif 'thousand' in match_text or 'lakh' in context_window:
                        if value < 10000:
                            value *= 1_000
                    elif 'crore' in match_text or 'crore' in context_window:
                        if value < 100000:
                            value *= 10_000_000

                # Post-processing normalization: check sanity bounds
                normalized = normalize_metric_value(value, metric_name)
                if normalized is None:
                    continue  # Value out of reasonable range — skip

                value = normalized

                # Convert to int if whole number
                if isinstance(value, float) and value.is_integer() and value < 1e15:
                    value = int(value)

                return {
                    'value': value,
                    'unit': metric_def['unit'],
                    'raw_match': match.group(0),
                }
            except (ValueError, IndexError):
                continue

    return None



def _calculate_ml_confidence(
    model_confidence: float,
    has_value: bool,
    has_keywords: bool,
    metric_name: str,
    text: str,
) -> float:
    """
    Calculate a combined confidence score incorporating model prediction
    and extraction heuristics.
    """
    confidence = model_confidence * 0.6  # Model confidence contributes 60%

    if has_value:
        confidence += 0.2  # Found a numeric value

    if has_keywords:
        confidence += 0.1  # Keywords present

    # Check for ESG context terms
    esg_context = ['report', 'sustainability', 'esg', 'metric', 'performance',
                   'target', 'disclosure', 'brsr', 'annual']
    text_lower = text.lower()
    context_count = sum(1 for term in esg_context if term in text_lower)
    confidence += min(context_count * 0.02, 0.1)

    # Penalize uncertainty
    uncertainty = ['approximately', 'about', 'estimated', 'around', 'nearly']
    if any(w in text_lower for w in uncertainty):
        confidence -= 0.05

    return round(min(max(confidence, 0.0), 1.0), 3)


# ─── MAIN EXTRACTION PIPELINE ────────────────────────────────────────────────

class MLESGExtractor:
    """
    Full ML-based ESG metric extraction pipeline.
    
    Uses RoBERTa with multi-head classifier to classify text chunks,
    then regex to extract values. Falls back to pure regex when 
    model confidence is below threshold.
    
    Hybrid approach: Final output = ML predictions + Regex fallback merge
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        Initialize the ML extractor.
        
        Args:
            model_path: Path to saved model checkpoint. If None, uses untrained model + regex fallback.
            confidence_threshold: Minimum ML confidence to accept a prediction (lowered to 0.35 for better recall)
            device: 'cuda', 'cpu', or None for auto-detect
        """
        # Set device
        if device is None:
            self.device = torch.device('cpu' )
        else:
            self.device = torch.device(device)

        print(f"  [MLExtractor] Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = 512
        self.confidence_threshold = confidence_threshold

        # Load or initialize model
        self.model = ESGMetricClassifier(num_classes=NUM_CLASSES)
        self.model_loaded = False

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("  [MLExtractor] No trained model found. Using regex-only fallback mode.")

        self.model.to(self.device)

    def _load_model(self, model_path: str):
        """Load a trained model checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_loaded = True
            print(f"  [MLExtractor] Model loaded from {model_path}")
            if 'metrics' in checkpoint:
                print(f"  [MLExtractor] Training metrics: {checkpoint['metrics']}")
        except Exception as e:
            print(f"  [MLExtractor] ERROR loading model: {e}")
            self.model_loaded = False

    def _tokenize(self, texts: List[str]) -> Dict:
        """Tokenize a batch of texts."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _regex_fallback_extract(self, text: str) -> Dict[str, Dict]:
        """
        Pure regex extraction as fallback when ML model is unavailable or low confidence.
        """
        results = {}
        for metric_name, metric_def in ESG_METRICS.items():
            extraction = extract_value_and_unit(text, metric_name)
            if extraction is not None:
                # Check keyword presence for confidence
                has_keywords = any(kw in text.lower() for kw in metric_def['keywords'])
                confidence = 0.6 if has_keywords else 0.4
                if ':' in extraction['raw_match'] or '=' in extraction['raw_match']:
                    confidence += 0.1
                
                results[metric_name] = {
                    'value': extraction['value'],
                    'unit': extraction['unit'],
                    'confidence': round(min(confidence + 0.1, 1.0), 3),
                    'source': 'regex',
                    'context': extraction['raw_match'],
                }
        return results

    def extract_from_text(self, text: str, chunk_size: int = 256, overlap: int = 64) -> Dict[str, Dict]:
        """
        Extract ESG metrics from a text string.
        
        Hybrid extraction pipeline:
        1. Chunks the text (smaller chunks for better localization)
        2. Classifies each chunk with the ML model (if available)
        3. Extracts values using regex for identified metrics
        4. Falls back to regex-only for low-confidence predictions
        5. Merges ML + regex results (highest confidence wins)
        6. Applies post-processing normalization
        
        Args:
            text: Full text to extract from
            chunk_size: Words per chunk (default: 256 for better ESG metric localization)
            overlap: Overlap words (default: 64)
            
        Returns:
            Dict mapping metric_name → {value, unit, confidence, source, context}
        """
        if not text or not text.strip():
            return {}

        # Clean the text first
        cleaned = clean_text(text)

        # If no trained model, use regex fallback
        if not self.model_loaded:
            return self._regex_fallback_extract(cleaned)

        # Chunk the text
        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return self._regex_fallback_extract(cleaned)

        # Classify chunks in batches
        all_results = {}
        batch_size = 4  # Smaller batch for RoBERTa on MX550 2GB VRAM
        chunk_texts = [c['text'] for c in chunks]

        self.model.eval()

        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i:i + batch_size]
            encoded = self._tokenize(batch_texts)

            metric_names, confidences = self.model.predict(
                encoded['input_ids'], encoded['attention_mask']
            )

            for j, (pred_metric, conf) in enumerate(zip(metric_names, confidences)):
                chunk_text_str = batch_texts[j]

                # Skip 'no_metric' predictions
                if pred_metric == 'no_metric':
                    continue

                # Skip low-confidence predictions (threshold lowered to 0.35)
                if conf < self.confidence_threshold:
                    continue

                # Try to extract value for the predicted metric
                extraction = extract_value_and_unit(chunk_text_str, pred_metric)

                if extraction is not None:
                    has_keywords = any(
                        kw in chunk_text_str.lower()
                        for kw in ESG_METRICS[pred_metric]['keywords']
                    )
                    final_confidence = _calculate_ml_confidence(
                        conf, True, has_keywords, pred_metric, chunk_text_str
                    )

                    # Keep the result with highest confidence
                    if pred_metric not in all_results or \
                       final_confidence > all_results[pred_metric]['confidence']:
                        all_results[pred_metric] = {
                            'value': extraction['value'],
                            'unit': extraction['unit'],
                            'confidence': final_confidence,
                            'source': 'ml',
                            'context': extraction['raw_match'],
                        }

        # HYBRID MERGE: Augment with regex fallback for metrics ML missed
        # This is the key boost — regex catches what ML misses
        regex_results = self._regex_fallback_extract(cleaned)
        for metric_name, regex_result in regex_results.items():
            if metric_name not in all_results:
                # ML missed this metric entirely → include regex result
                all_results[metric_name] = regex_result
            elif all_results[metric_name]['confidence'] < regex_result['confidence']:
                # Regex found a higher confidence match
                all_results[metric_name] = regex_result

        return all_results


def ml_esg_metric_extractor(
    text: str,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.25,
) -> Dict[str, Dict]:
    """
    Convenience function for one-shot ESG metric extraction.
    
    Args:
        text: Full text to extract metrics from
        model_path: Optional path to trained model
        confidence_threshold: Minimum confidence to accept (default: 0.35)
        
    Returns:
        Dict of extracted metrics {metric_name: {value, unit, confidence}}
    """
    extractor = MLESGExtractor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )
    return extractor.extract_from_text(text)


if __name__ == "__main__":
    # Quick test with sample text
    test_text = """
    Our company's total greenhouse gas emissions were 125,000 tCO2e in FY2023-24.
    Scope 1 emissions: 45,000 tCO2e (direct emissions from owned sources)
    Scope 2 emissions: 35,000 tCO2e (indirect emissions from purchased electricity)
    Scope 3 emissions: 45,000 tCO2e (value chain emissions)
    
    Energy consumption was 85,000 MWh, with 42% from renewable sources.
    Water withdrawal: 2,500,000 m3.
    62% of waste was recycled.
    
    Employee turnover rate: 12%
    Female representation in workforce: 36%
    Average training hours per employee: 24 hours
    
    Board independence: 55%
    Female directors: 25%
    CEO pay ratio: 85:1
    Ethics training completion: 92%
    """

    print("=" * 60)
    print("ML ESG METRIC EXTRACTOR - Test")
    print("=" * 60)

    results = ml_esg_metric_extractor(test_text)

    print(f"\nExtracted {len(results)} metrics:")
    for metric, info in sorted(results.items()):
        print(f"  {metric:25s}: {info['value']} {info['unit']} "
              f"(confidence: {info['confidence']:.2f}, source: {info['source']})")
