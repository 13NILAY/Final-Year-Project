"""
Evaluation Module
=================
Evaluate ML extraction accuracy and compare with regex baseline.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .labeling import ESG_METRICS, METRIC_NAMES, METRIC_TO_ID, ID_TO_METRIC, CATEGORY_MAP


def evaluate_extraction(
    predictions: Dict[str, Dict],
    ground_truths: Dict[str, Dict],
    value_tolerance: float = 0.15,
) -> Dict:
    """
    Evaluate extraction predictions against ground truth.
    
    Args:
        predictions: {metric_name: {value, unit, confidence}}
        ground_truths: {metric_name: {value, unit}}
        value_tolerance: Acceptable relative error for value matching (0.15 = 15%)
        
    Returns:
        Evaluation metrics dict
    """
    all_metrics = set(list(predictions.keys()) + list(ground_truths.keys()))

    tp = 0  # Correctly extracted
    fp = 0  # Extracted but wrong
    fn = 0  # Missed (in ground truth but not extracted)
    value_matches = 0
    total_with_values = 0

    detailed = []

    for metric in all_metrics:
        pred = predictions.get(metric)
        truth = ground_truths.get(metric)

        if truth is not None and pred is not None:
            # Both exist - check value accuracy
            tp += 1
            total_with_values += 1

            pred_val = pred.get('value', 0)
            truth_val = truth.get('value', 0)

            if truth_val != 0:
                rel_error = abs(pred_val - truth_val) / abs(truth_val)
                value_match = rel_error <= value_tolerance
            else:
                value_match = pred_val == 0

            if value_match:
                value_matches += 1

            detailed.append({
                'metric': metric,
                'status': 'correct' if value_match else 'value_mismatch',
                'predicted_value': pred_val,
                'truth_value': truth_val,
                'confidence': pred.get('confidence', 0),
                'source': pred.get('source', 'unknown'),
            })

        elif pred is not None and truth is None:
            fp += 1
            detailed.append({
                'metric': metric,
                'status': 'false_positive',
                'predicted_value': pred.get('value', 0),
                'truth_value': None,
                'confidence': pred.get('confidence', 0),
                'source': pred.get('source', 'unknown'),
            })

        elif truth is not None and pred is None:
            fn += 1
            detailed.append({
                'metric': metric,
                'status': 'missed',
                'predicted_value': None,
                'truth_value': truth.get('value', 0),
                'confidence': 0,
                'source': 'none',
            })

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    value_accuracy = value_matches / total_with_values if total_with_values > 0 else 0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'value_accuracy': round(value_accuracy, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_ground_truth': len(ground_truths),
        'total_predicted': len(predictions),
        'detailed': detailed,
    }


def evaluate_by_category(
    predictions: Dict[str, Dict],
    ground_truths: Dict[str, Dict],
    value_tolerance: float = 0.15,
) -> Dict:
    """
    Evaluate extraction grouped by ESG category (Environmental, Social, Governance).
    """
    results = {}

    for category, metrics in CATEGORY_MAP.items():
        cat_preds = {k: v for k, v in predictions.items() if k in metrics}
        cat_truths = {k: v for k, v in ground_truths.items() if k in metrics}
        results[category] = evaluate_extraction(cat_preds, cat_truths, value_tolerance)

    # Overall
    results['overall'] = evaluate_extraction(predictions, ground_truths, value_tolerance)

    return results


def compare_with_regex(
    ml_results: Dict[str, Dict],
    regex_results: Dict[str, Dict],
    ground_truth: Dict[str, Dict],
    value_tolerance: float = 0.15,
) -> Dict:
    """
    Compare ML extraction results with regex baseline.
    
    Returns:
        Comparison dict with per-method and per-category scores
    """
    ml_eval = evaluate_by_category(ml_results, ground_truth, value_tolerance)
    regex_eval = evaluate_by_category(regex_results, ground_truth, value_tolerance)

    comparison = {
        'ml': ml_eval,
        'regex': regex_eval,
        'improvement': {},
    }

    # Calculate improvements
    for category in ['environmental', 'social', 'governance', 'overall']:
        ml_f1 = ml_eval.get(category, {}).get('f1', 0)
        regex_f1 = regex_eval.get(category, {}).get('f1', 0)
        improvement = ml_f1 - regex_f1

        comparison['improvement'][category] = {
            'ml_f1': ml_f1,
            'regex_f1': regex_f1,
            'f1_improvement': round(improvement, 4),
            'better': 'ml' if improvement > 0 else ('regex' if improvement < 0 else 'tie'),
        }

    return comparison


def generate_evaluation_report(
    eval_results: Dict,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a human-readable evaluation report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ESG METRIC EXTRACTION - EVALUATION REPORT")
    lines.append("=" * 60)

    if 'overall' in eval_results:
        overall = eval_results['overall']
        lines.append(f"\n📊 OVERALL METRICS:")
        lines.append(f"  Precision:      {overall['precision']:.2%}")
        lines.append(f"  Recall:         {overall['recall']:.2%}")
        lines.append(f"  F1 Score:       {overall['f1']:.2%}")
        lines.append(f"  Value Accuracy: {overall['value_accuracy']:.2%}")
        lines.append(f"  TP: {overall['tp']} | FP: {overall['fp']} | FN: {overall['fn']}")

    for category in ['environmental', 'social', 'governance']:
        if category in eval_results:
            cat = eval_results[category]
            emoji = {'environmental': '🌍', 'social': '👥', 'governance': '🏛️'}
            lines.append(f"\n{emoji.get(category, '')} {category.upper()}:")
            lines.append(f"  Precision: {cat['precision']:.2%} | "
                        f"Recall: {cat['recall']:.2%} | "
                        f"F1: {cat['f1']:.2%}")
            lines.append(f"  Value Accuracy: {cat['value_accuracy']:.2%}")

    # Detailed per-metric results
    if 'overall' in eval_results and 'detailed' in eval_results['overall']:
        lines.append(f"\n📋 PER-METRIC DETAILS:")
        for d in eval_results['overall']['detailed']:
            status_emoji = {
                'correct': '✅',
                'value_mismatch': '⚠️',
                'false_positive': '❌',
                'missed': '🔍',
            }
            emoji = status_emoji.get(d['status'], '  ')
            lines.append(f"  {emoji} {d['metric']:25s} | "
                        f"Status: {d['status']:15s} | "
                        f"Pred: {d['predicted_value']} | "
                        f"Truth: {d['truth_value']} | "
                        f"Conf: {d['confidence']:.2f}")

    report = '\n'.join(lines)
    print(report)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w',encoding='utf-8') as f:
            f.write(report)
        print(f"\n[Evaluation] Report saved to {output_path}")

    return report


def generate_comparison_report(
    comparison: Dict,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a comparison report between ML and regex extraction.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ML vs REGEX EXTRACTION - COMPARISON REPORT")
    lines.append("=" * 60)

    lines.append(f"\n{'Category':<20} {'ML F1':>10} {'Regex F1':>10} {'Improvement':>12} {'Winner':>8}")
    lines.append("-" * 60)

    for category in ['environmental', 'social', 'governance', 'overall']:
        imp = comparison['improvement'].get(category, {})
        ml_f1 = imp.get('ml_f1', 0)
        regex_f1 = imp.get('regex_f1', 0)
        improvement = imp.get('f1_improvement', 0)
        better = imp.get('better', 'tie')

        icon = '🟢' if better == 'ml' else ('🔴' if better == 'regex' else '⚪')
        cat_display = category.upper() if category == 'overall' else category.capitalize()
        lines.append(f"  {icon} {cat_display:<18} {ml_f1:>10.2%} {regex_f1:>10.2%} "
                    f"{improvement:>+10.2%} {better:>8}")

    report = '\n'.join(lines)
    print(report)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w',encoding='utf-8') as f:
            f.write(report)

    return report


# ─── TEST SAMPLES FOR VALIDATION ─────────────────────────────────────────────

DUMMY_TEST_SAMPLES = [
    {
        'text': """Our total greenhouse gas emissions were 125,000 tCO2e. 
                   Scope 1: 45,000 tCO2e, Scope 2: 35,000 tCO2e, Scope 3: 45,000 tCO2e.
                   Renewable energy usage: 42%. Energy consumption: 85,000 MWh.""",
        'ground_truth': {
            'ghg_emissions': {'value': 125000, 'unit': 'tCO2e'},
            'scope1_emissions': {'value': 45000, 'unit': 'tCO2e'},
            'scope2_emissions': {'value': 35000, 'unit': 'tCO2e'},
            'scope3_emissions': {'value': 45000, 'unit': 'tCO2e'},
            'renewable_energy': {'value': 42, 'unit': '%'},
            'energy_consumption': {'value': 85000, 'unit': 'MWh'},
        }
    },
    {
        'text': """Employee turnover rate: 12%. Female representation: 36%.
                   Average training hours per employee: 24 hours.
                   Board independence: 55%. Female directors: 25%.""",
        'ground_truth': {
            'employee_turnover': {'value': 12, 'unit': '%'},
            'female_representation': {'value': 36, 'unit': '%'},
            'training_hours': {'value': 24, 'unit': 'hours'},
            'board_independence': {'value': 55, 'unit': '%'},
            'female_directors': {'value': 25, 'unit': '%'},
        }
    },
    {
        'text': """The company invested INR 150 crore in community investment programs.
                   CEO pay ratio: 85:1. Ethics training completion: 92%.
                   Whistleblower cases reported: 12. Lost time injury rate: 0.35.""",
        'ground_truth': {
            'community_investment': {'value': 150, 'unit': 'INR Crore'},
            'ceo_pay_ratio': {'value': 85, 'unit': 'ratio'},
            'ethics_training': {'value': 92, 'unit': '%'},
            'whistleblower_cases': {'value': 12, 'unit': 'count'},
            'lost_time_injury': {'value': 0.35, 'unit': 'rate'},
        }
    },
    {
        'text': """Water withdrawal totaled 2,500,000 m3. 
                   Hazardous waste generated: 450 tonnes. 
                   Waste recycling rate: 62%.
                   CO2 emissions: 110,000 tCO2e.""",
        'ground_truth': {
            'water_withdrawal': {'value': 2500000, 'unit': 'm3'},
            'hazardous_waste': {'value': 450, 'unit': 'tonnes'},
            'waste_recycled': {'value': 62, 'unit': '%'},
            'co2_emissions': {'value': 110000, 'unit': 'tCO2e'},
        }
    },
    {
        'text': """Employee satisfaction survey results showed 78% satisfaction.
                   Scope 1 emissions reduced to 28,000 tCO2e, representing a 15% decrease.
                   55% of our energy comes from renewable sources.""",
        'ground_truth': {
            'employee_satisfaction': {'value': 78, 'unit': '%'},
            'scope1_emissions': {'value': 28000, 'unit': 'tCO2e'},
            'renewable_energy': {'value': 55, 'unit': '%'},
        }
    },
]


if __name__ == "__main__":
    from .extractor import MLESGExtractor

    print("=" * 60)
    print("ESG EXTRACTION - EVALUATION")
    print("=" * 60)

    extractor = MLESGExtractor()

    all_preds = {}
    all_truths = {}

    for i, sample in enumerate(DUMMY_TEST_SAMPLES):
        print(f"\n--- Test Sample {i+1} ---")
        preds = extractor.extract_from_text(sample['text'])
        truths = sample['ground_truth']

        for k, v in preds.items():
            all_preds[f"s{i}_{k}"] = v
        for k, v in truths.items():
            all_truths[f"s{i}_{k}"] = v

        result = evaluate_extraction(preds, truths)
        print(f"  F1: {result['f1']:.2%} | Precision: {result['precision']:.2%} | "
              f"Recall: {result['recall']:.2%}")

    # Overall evaluation
    print("\n")
    overall = evaluate_by_category(
        {k.split('_', 1)[1]: v for k, v in all_preds.items()},
        {k.split('_', 1)[1]: v for k, v in all_truths.items()},
    )
    generate_evaluation_report(overall)
