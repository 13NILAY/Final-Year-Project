"""
Evaluation Module (Improved)
============================
Evaluate extraction accuracy with rich metrics including qualifiers,
value tolerance, and error categorization.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime

from .canonical_metrics import get_alias_manager
from .extractor import ESGStagedExtractor


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------

def _value_match(val1: Optional[float], val2: Optional[float], tolerance: float = 0.15) -> bool:
    """Check if two numeric values match within relative tolerance."""
    if val1 is None or val2 is None:
        return False
    if val1 == 0 and val2 == 0:
        return True
    if val1 == 0 or val2 == 0:
        return False
    return abs(val1 - val2) / max(abs(val1), abs(val2)) <= tolerance

def _unit_match(unit1: str, unit2: str, alias_manager) -> bool:
    """Normalize and compare units."""
    # Simple normalization: lowercase, strip
    u1 = unit1.lower().strip() if unit1 else ''
    u2 = unit2.lower().strip() if unit2 else ''
    # If both empty, consider match
    if not u1 and not u2:
        return True
    # Could add unit aliases later
    return u1 == u2


def _qualifier_match(q1: Any, q2: Any) -> bool:
    """Match qualifiers (year, scope, etc.) - allow None to match anything."""
    if q1 is None or q2 is None:
        return True  # missing qualifier is not penalized
    return q1 == q2


# ----------------------------------------------------------------------
# Main evaluation function
# ----------------------------------------------------------------------

def evaluate_extraction(
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    value_tolerance: float = 0.15,
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth at the sample level.
    Each sample is a dict with fields: metric_name, value, unit, year, scope,
    actual_or_target, geography, entity_level, source_text (optional), etc.
    Returns detailed evaluation metrics.
    """
    alias_manager = get_alias_manager(use_embeddings=False)

    # Metrics per canonical metric
    per_metric = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0,
                                       'value_correct': 0, 'unit_correct': 0,
                                       'qualifier_correct': defaultdict(int)})
    
    # Error buckets
    error_buckets = defaultdict(int)

    # Track matched predictions to avoid double-counting
    matched_preds = set()

    # For each ground truth sample, find best matching prediction
    for gt in ground_truth:
        gt_metric = gt['metric_name']
        if gt_metric == 'no_metric':
            # Skip negative samples? We'll handle separately.
            continue

        best_match = None
        best_score = -1  # higher is better

        for i, pred in enumerate(predictions):
            if i in matched_preds:
                continue
            if pred['canonical_metric'] != gt_metric:
                continue  # metric mismatch -> not a candidate for this GT

            # Compute match quality: value, unit, qualifiers
            score = 0
            if _value_match(pred['value'], gt['value'], value_tolerance):
                score += 10
            if _unit_match(pred.get('normalized_unit', pred['raw_unit']), gt['unit'], alias_manager):
                score += 5
            if _qualifier_match(pred.get('year'), gt.get('year')):
                score += 2
            if _qualifier_match(pred.get('scope'), gt.get('scope')):
                score += 2
            if _qualifier_match(pred.get('actual_or_target'), gt.get('actual_or_target')):
                score += 1
            if _qualifier_match(pred.get('geography'), gt.get('geography')):
                score += 1
            if _qualifier_match(pred.get('entity_level'), gt.get('entity_level')):
                score += 1

            if score > best_score:
                best_score = score
                best_match = (i, pred, score)

        if best_match:
            idx, pred, _ = best_match
            matched_preds.add(idx)

            # Update per-metric stats
            pm = per_metric[gt_metric]
            pm['tp'] += 1

            # Value correctness
            if _value_match(pred['value'], gt['value'], value_tolerance):
                pm['value_correct'] += 1
            else:
                error_buckets['value_mismatch'] += 1

            # Unit correctness
            if _unit_match(pred.get('normalized_unit', pred['raw_unit']), gt['unit'], alias_manager):
                pm['unit_correct'] += 1
            else:
                error_buckets['unit_mismatch'] += 1

            # Qualifier correctness
            for q in ['year', 'scope', 'actual_or_target', 'geography', 'entity_level']:
                if _qualifier_match(pred.get(q), gt.get(q)):
                    pm['qualifier_correct'][q] += 1
                else:
                    error_buckets[f'{q}_mismatch'] += 1

            # Exact match (all fields correct)
            exact = (
                _value_match(pred['value'], gt['value'], value_tolerance) and
                _unit_match(pred.get('normalized_unit', pred['raw_unit']), gt['unit'], alias_manager) and
                all(_qualifier_match(pred.get(q), gt.get(q)) for q in ['year', 'scope', 'actual_or_target', 'geography', 'entity_level'])
            )
            if exact:
                pm['exact'] = pm.get('exact', 0) + 1
            else:
                error_buckets['partial_match'] += 1

        else:
            # No prediction for this GT metric
            per_metric[gt_metric]['fn'] += 1
            error_buckets['missed'] += 1

    # False positives: predictions not matched to any GT
    for i, pred in enumerate(predictions):
        if i not in matched_preds:
            pm = per_metric[pred['canonical_metric']]
            pm['fp'] += 1
            error_buckets['false_positive'] += 1

    # Compute aggregate metrics
    total_tp = sum(pm['tp'] for pm in per_metric.values())
    total_fp = sum(pm['fp'] for pm in per_metric.values())
    total_fn = sum(pm['fn'] for pm in per_metric.values())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Value/unit/qualifier accuracy among correct detections
    value_acc = sum(pm['value_correct'] for pm in per_metric.values()) / total_tp if total_tp > 0 else 0
    unit_acc = sum(pm['unit_correct'] for pm in per_metric.values()) / total_tp if total_tp > 0 else 0
    qualifier_acc = {}
    for q in ['year', 'scope', 'actual_or_target', 'geography', 'entity_level']:
        qualifier_acc[q] = sum(pm['qualifier_correct'][q] for pm in per_metric.values()) / total_tp if total_tp > 0 else 0

    exact_match = sum(pm.get('exact', 0) for pm in per_metric.values()) / total_tp if total_tp > 0 else 0

    # Per-metric breakdown
    per_metric_summary = {}
    for metric, pm in per_metric.items():
        p = pm['tp'] / (pm['tp'] + pm['fp']) if (pm['tp'] + pm['fp']) > 0 else 0
        r = pm['tp'] / (pm['tp'] + pm['fn']) if (pm['tp'] + pm['fn']) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_metric_summary[metric] = {
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f, 4),
            'tp': pm['tp'],
            'fp': pm['fp'],
            'fn': pm['fn'],
            'value_accuracy': pm['value_correct'] / pm['tp'] if pm['tp'] > 0 else 0,
            'unit_accuracy': pm['unit_correct'] / pm['tp'] if pm['tp'] > 0 else 0,
        }

    return {
        'overall': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'value_accuracy': round(value_acc, 4),
            'unit_accuracy': round(unit_acc, 4),
            'exact_match': round(exact_match, 4),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
        },
        'qualifier_accuracy': {k: round(v, 4) for k, v in qualifier_acc.items()},
        'per_metric': per_metric_summary,
        'error_buckets': dict(error_buckets),
    }


# ----------------------------------------------------------------------
# Evaluation on test dataset
# ----------------------------------------------------------------------

def evaluate_on_testset(
    test_path: str,
    model_path: Optional[str] = None,
    value_tolerance: float = 0.15,
) -> Dict[str, Any]:
    """
    Load test dataset, run extraction on each sample's text, and evaluate.
    """
    # Load test samples
    ground_truth = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            # Only include positive samples for evaluation (skip no_metric)
            if sample.get('metric_name') != 'no_metric':
                ground_truth.append(sample)

    # Initialize extractor
    extractor = ESGStagedExtractor(model_path=model_path)

    # Run extraction on each sample's text (treat each chunk independently)
    predictions = []
    for gt in ground_truth:
        text = gt['text']
        # Extract from this chunk
        rich = extractor.extract_from_text(text)
        # For each predicted metric, create a prediction record
        for canonical, info in rich.items():
            if info['value'] is None:
                continue   # skip predictions that didn't extract a numeric value
            pred = {
                'canonical_metric': canonical,
                'value': info['value'],
                'raw_unit': info['raw_unit'],
                'normalized_unit': info['normalized_unit'],
                'year': info.get('year'),
                'scope': info.get('scope'),
                'actual_or_target': info.get('actual_or_target'),
                'geography': info.get('geography'),
                'entity_level': info.get('entity_level'),
                'confidence': info['confidence'],
            }
            predictions.append(pred)

    # Evaluate
    results = evaluate_extraction(ground_truth, predictions, value_tolerance)
    return results


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------

def generate_evaluation_report(
    eval_results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a human-readable evaluation report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ESG METRIC EXTRACTION - DETAILED EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    overall = eval_results['overall']
    lines.append("\n📊 OVERALL METRICS:")
    lines.append(f"  Precision:      {overall['precision']:.2%}")
    lines.append(f"  Recall:         {overall['recall']:.2%}")
    lines.append(f"  F1 Score:       {overall['f1']:.2%}")
    lines.append(f"  Value Accuracy: {overall['value_accuracy']:.2%}")
    lines.append(f"  Unit Accuracy:  {overall['unit_accuracy']:.2%}")
    lines.append(f"  Exact Match:    {overall['exact_match']:.2%}")
    lines.append(f"  TP: {overall['tp']} | FP: {overall['fp']} | FN: {overall['fn']}")

    lines.append("\n🎯 QUALIFIER ACCURACY (among correct detections):")
    for q, acc in eval_results['qualifier_accuracy'].items():
        lines.append(f"  {q:18s}: {acc:.2%}")

    lines.append("\n🔍 ERROR BUCKETS:")
    for err, count in eval_results['error_buckets'].items():
        lines.append(f"  {err:25s}: {count}")

    lines.append("\n📈 PER-METRIC BREAKDOWN:")
    lines.append(f"{'Metric':30s} {'P':>6} {'R':>6} {'F1':>6} {'Val%':>6} {'Unit%':>6}")
    lines.append("-" * 70)
    for metric, pm in sorted(eval_results['per_metric'].items()):
        lines.append(f"{metric:30s} {pm['precision']:.2%} {pm['recall']:.2%} "
                     f"{pm['f1']:.2%} {pm['value_accuracy']:.2%} {pm['unit_accuracy']:.2%}")

    report = '\n'.join(lines)
    print(report)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[Evaluation] Report saved to {output_path}")

    return report


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ESG extraction.")
    parser.add_argument("test_path", help="Path to test JSONL file (rich format)")
    parser.add_argument("--model", help="Path to model checkpoint (optional)")
    parser.add_argument("--output", help="Output report file (optional)")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Value tolerance (default 0.15)")
    args = parser.parse_args()

    results = evaluate_on_testset(
        test_path=args.test_path,
        model_path=args.model,
        value_tolerance=args.tolerance,
    )

    generate_evaluation_report(results, args.output)

    # Also save raw results as JSON
    if args.output:
        json_output = args.output.replace('.txt', '.json') if args.output.endswith('.txt') else args.output + '.json'
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Raw results saved to {json_output}")