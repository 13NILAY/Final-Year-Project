"""
Pipeline Orchestrator
=====================
Unified pipeline: PDF → preprocess → ML extract → score → results.
Drop-in replacement for ESGReportAnalyzer.analyze_report().
"""

import os
import sys
import json
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add parent directory to path so we can import esg_new
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .preprocessing import extract_text_from_pdf, clean_text, preprocess_pdf
from .extractor import MLESGExtractor
from .labeling import ESG_METRICS, CATEGORY_MAP, generate_labeled_dataset, load_dataset


class ESGMLPipeline:
    """
    Complete ML-based ESG analysis pipeline.
    
    This is a drop-in replacement for the regex-based ESGReportAnalyzer.
    It uses a fine-tuned RoBERTa model with multi-head classifier for
    metric identification with regex fallback for value extraction.
    
    Usage:
        pipeline = ESGMLPipeline(industry='technology')
        results = pipeline.analyze_report('path/to/esg_report.pdf')
    """

    # Industry-specific score thresholds (matching esg_new.py ESGScoreRanges)
    ENVIRONMENTAL_RANGES = {
        'ghg_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.15, 'threshold': 100000},
        'scope1_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 50000},
        'scope2_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 50000},
        'scope3_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 200000},
        'co2_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 100000},
        'energy_consumption': {'unit': 'MWh', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 500000},
        'renewable_energy': {'unit': '%', 'ideal': (40, 100), 'weight': 0.15},
        'water_withdrawal': {'unit': 'm3', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 5000000},
        'waste_recycled': {'unit': '%', 'ideal': (50, 100), 'weight': 0.05},
        'hazardous_waste': {'unit': 'tonnes', 'ideal': 'decreasing', 'weight': 0.05, 'threshold': 1000},
    }

    SOCIAL_RANGES = {
        'employee_turnover': {'unit': '%', 'ideal': (5, 15), 'weight': 0.20},
        'female_representation': {'unit': '%', 'ideal': (30, 50), 'weight': 0.20},
        'training_hours': {'unit': 'hours', 'ideal': (20, 100), 'weight': 0.15},
        'lost_time_injury': {'unit': 'rate', 'ideal': (0, 1), 'weight': 0.15},
        'employee_satisfaction': {'unit': '%', 'ideal': (70, 100), 'weight': 0.15},
        'community_investment': {'unit': 'INR Crore', 'ideal': 'increasing', 'weight': 0.15, 'threshold': 100},
    }

    GOVERNANCE_RANGES = {
        'board_independence': {'unit': '%', 'ideal': (50, 100), 'weight': 0.25},
        'female_directors': {'unit': '%', 'ideal': (25, 50), 'weight': 0.20},
        'ceo_pay_ratio': {'unit': 'ratio', 'ideal': (1, 50), 'weight': 0.25},
        'ethics_training': {'unit': '%', 'ideal': (90, 100), 'weight': 0.15},
        'whistleblower_cases': {'unit': 'count', 'ideal': (0, 5), 'weight': 0.15},
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        industry: str = 'general',
        confidence_threshold: float = 0.35,
        device: Optional[str] = "cuda",
    ):
        """
        Initialize the ML pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            industry: Industry for scoring norms ('general', 'technology', 'manufacturing', etc.)
            confidence_threshold: Min ML confidence to accept prediction
            device: 'cuda', 'cpu', or None for auto-detect
        """
        self.industry = industry.lower()
        self.confidence_threshold = confidence_threshold

        # Auto-detect model path if not provided
        if model_path is None:
            default_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "models", "best_model.pt"
            )
            if os.path.exists(default_path):
                model_path = default_path

        self.extractor = MLESGExtractor(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

        print(f"  [Pipeline] Industry: {self.industry}")
        print(f"  [Pipeline] ML model: {'loaded' if self.extractor.model_loaded else 'regex-only mode'}")

    def analyze_report(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Complete analysis of an ESG report.
        Drop-in replacement for ESGReportAnalyzer.analyze_report().
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Analysis results dict matching existing system format
        """
        print(f"\n{'='*60}")
        print(f"ML PIPELINE: ANALYZING ESG REPORT")
        print(f"File: {os.path.basename(pdf_path)}")
        print(f"Industry: {self.industry.title()}")
        print(f"{'='*60}")

        # Step 1: Extract text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print("❌ Failed to extract text from PDF.")
            return None

        print(f"✓ Extracted {len(raw_text):,} characters from PDF")

        # Step 2: Extract metrics using ML + regex
        extracted_metrics = self.extractor.extract_from_text(raw_text)
        print(f"✓ Found {len(extracted_metrics)} ESG metrics")

        if not extracted_metrics:
            print("⚠ No ESG metrics found in this report.")

        # Step 3: Calculate ESG scores
        scores = self._calculate_category_scores(extracted_metrics)

        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(extracted_metrics, scores)

        # Step 5: Prepare results (matching existing format)
        results = {
            'company': os.path.basename(pdf_path).replace('.pdf', ''),
            'industry': self.industry,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'extracted_metrics': extracted_metrics,
            'esg_scores': scores,
            'recommendations': recommendations,
            'analysis_summary': self._create_summary(scores, extracted_metrics, recommendations),
            'pipeline': 'ml',  # Flag to identify ML pipeline results
        }

        return results

    def batch_analyze(self, pdf_dir: str) -> List[Dict]:
        """Analyze all PDFs in a directory."""
        results = []
        pdf_files = []

        for root, dirs, files in os.walk(pdf_dir):
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, f))

        print(f"\n[Pipeline] Batch analyzing {len(pdf_files)} PDFs...")

        for pdf_path in pdf_files:
            try:
                result = self.analyze_report(pdf_path)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  ERROR processing {pdf_path}: {e}")

        print(f"[Pipeline] Successfully analyzed {len(results)}/{len(pdf_files)} reports")
        return results

    def _calculate_metric_score(self, metric_name: str, value: float) -> tuple:
        """Calculate score (0-100) for a single metric."""
        # Find the metric in the ranges
        for ranges in [self.ENVIRONMENTAL_RANGES, self.SOCIAL_RANGES, self.GOVERNANCE_RANGES]:
            if metric_name in ranges:
                info = ranges[metric_name]
                ideal = info['ideal']

                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    return 50, "Cannot score non-numeric value"

                if ideal == 'decreasing':
                    threshold = info.get('threshold', 1000)
                    if numeric_value <= 0:
                        score = 100
                    else:
                        ratio = min(numeric_value / threshold, 2.0)
                        score = max(0, 100 - (ratio * 50))
                    explanation = f"Current: {numeric_value:,.0f} {info['unit']}. Lower is better."

                elif ideal == 'increasing':
                    threshold = info.get('threshold', 1000000)
                    if numeric_value >= threshold:
                        score = 100
                    else:
                        score = min(100, (numeric_value / threshold) * 100)
                    explanation = f"Current: {numeric_value:,.0f} {info['unit']}. Higher is better."

                elif isinstance(ideal, tuple):
                    min_val, max_val = ideal
                    if min_val <= numeric_value <= max_val:
                        score = 100
                        explanation = f"Within ideal range ({min_val}-{max_val})."
                    elif numeric_value < min_val:
                        score = max(0, (numeric_value / min_val) * 100) if min_val > 0 else 0
                        explanation = f"Below ideal minimum ({min_val})."
                    else:
                        overshoot = (numeric_value - max_val) / max_val if max_val > 0 else 0
                        score = max(0, 100 - (overshoot * 100))
                        explanation = f"Above ideal maximum ({max_val})."
                else:
                    score = 50
                    explanation = "Scoring logic not defined"

                return round(min(100, max(0, score)), 1), explanation

        return 50, "Metric not in scoring framework"

    def _get_category(self, metric_name: str) -> str:
        """Get the ESG category for a metric."""
        for category, metrics in CATEGORY_MAP.items():
            if metric_name in metrics:
                return category
        return 'unknown'

    def _calculate_category_scores(self, extracted_metrics: Dict) -> Dict:
        """Calculate scores for each ESG category."""
        categories = {
            'environmental': {'metrics': [], 'weights': [], 'score': 0},
            'social': {'metrics': [], 'weights': [], 'score': 0},
            'governance': {'metrics': [], 'weights': [], 'score': 0},
        }

        ranges_map = {
            'environmental': self.ENVIRONMENTAL_RANGES,
            'social': self.SOCIAL_RANGES,
            'governance': self.GOVERNANCE_RANGES,
        }

        for metric_name, metric_info in extracted_metrics.items():
            category = self._get_category(metric_name)
            if category == 'unknown':
                continue

            value = metric_info.get('value', 0)
            score, explanation = self._calculate_metric_score(metric_name, value)

            weight = ranges_map.get(category, {}).get(metric_name, {}).get('weight', 0.1)

            categories[category]['metrics'].append({
                'metric': metric_name,
                'value': value,
                'unit': metric_info.get('unit', ''),
                'score': score,
                'weight': weight,
                'explanation': explanation,
                'confidence': metric_info.get('confidence', 0.5),
                'source': metric_info.get('source', 'unknown'),
            })
            categories[category]['weights'].append(weight)

        # Calculate weighted scores
        for category, data in categories.items():
            if data['metrics']:
                total_weight = sum(data['weights'])
                if total_weight > 0:
                    weighted_score = sum(
                        m['score'] * m['weight'] for m in data['metrics']
                    ) / total_weight
                    data['score'] = round(weighted_score, 1)
                else:
                    data['score'] = round(
                        sum(m['score'] for m in data['metrics']) / len(data['metrics']), 1
                    )
            else:
                data['score'] = 0

        # Overall score (weighted average of categories)
        env_score = categories['environmental']['score']
        soc_score = categories['social']['score']
        gov_score = categories['governance']['score']

        active = sum(1 for s in [env_score, soc_score, gov_score] if s > 0)
        if active > 0:
            overall = (env_score * 0.4 + soc_score * 0.3 + gov_score * 0.3)
            if active < 3:
                # Adjust if some categories have no data
                overall = sum(s for s in [env_score, soc_score, gov_score] if s > 0) / active
        else:
            overall = 0

        categories['overall'] = {'score': round(overall, 1)}

        return categories

    def _generate_recommendations(self, metrics: Dict, scores: Dict) -> List[Dict]:
        """Generate actionable recommendations based on extracted metrics and scores."""
        recommendations = []

        recommendation_templates = {
            'ghg_emissions': {
                'category': 'Environmental', 'priority': 'High', 'metric': 'GHG Emissions',
                'recommendation': 'Implement comprehensive carbon reduction strategy',
                'action_items': [
                    'Conduct detailed carbon footprint assessment across Scope 1, 2, and 3',
                    'Set science-based targets aligned with Paris Agreement',
                    'Invest in energy efficiency upgrades',
                    'Transition to renewable energy sources',
                ]
            },
            'scope1_emissions': {
                'category': 'Environmental', 'priority': 'High', 'metric': 'Scope 1 Emissions',
                'recommendation': 'Reduce direct emissions from owned sources',
                'action_items': [
                    'Upgrade equipment to energy-efficient models',
                    'Switch to low-carbon fuels',
                    'Implement fugitive emission controls',
                    'Electrify fleet vehicles',
                ]
            },
            'scope2_emissions': {
                'category': 'Environmental', 'priority': 'Medium', 'metric': 'Scope 2 Emissions',
                'recommendation': 'Reduce indirect emissions from purchased energy',
                'action_items': [
                    'Purchase Renewable Energy Certificates (RECs)',
                    'Enter Power Purchase Agreements (PPAs)',
                    'Install on-site renewable generation',
                    'Improve energy efficiency in facilities',
                ]
            },
            'scope3_emissions': {
                'category': 'Environmental', 'priority': 'Medium', 'metric': 'Scope 3 Emissions',
                'recommendation': 'Address value chain emissions',
                'action_items': [
                    'Engage suppliers on carbon reduction',
                    'Optimize logistics and transportation',
                    'Reduce business travel',
                    'Design products for lower carbon footprint',
                ]
            },
            'renewable_energy': {
                'category': 'Environmental', 'priority': 'Medium', 'metric': 'Renewable Energy',
                'recommendation': 'Increase renewable energy usage',
                'action_items': [
                    'Install on-site solar panels',
                    'Purchase Renewable Energy Certificates',
                    'Enter Power Purchase Agreements with renewable providers',
                    'Join industry initiatives like RE100',
                ]
            },
            'female_representation': {
                'category': 'Social', 'priority': 'High', 'metric': 'Gender Diversity',
                'recommendation': 'Enhance gender diversity and inclusion',
                'action_items': [
                    'Implement blind recruitment processes',
                    'Set and disclose diversity targets',
                    'Create leadership programs for women',
                    'Conduct regular pay equity audits',
                ]
            },
            'employee_turnover': {
                'category': 'Social', 'priority': 'Medium', 'metric': 'Employee Retention',
                'recommendation': 'Improve employee retention and satisfaction',
                'action_items': [
                    'Conduct stay interviews',
                    'Enhance career development opportunities',
                    'Improve work-life balance',
                    'Benchmark compensation against market',
                ]
            },
            'board_independence': {
                'category': 'Governance', 'priority': 'Medium', 'metric': 'Board Independence',
                'recommendation': 'Strengthen board independence and oversight',
                'action_items': [
                    'Increase proportion of independent directors',
                    'Separate CEO and Chair roles',
                    'Enhance board diversity',
                    'Implement regular board effectiveness reviews',
                ]
            },
        }

        # Generate recommendations for metrics with low scores
        for category_name, cat_data in scores.items():
            if category_name == 'overall':
                continue
            for metric_data in cat_data.get('metrics', []):
                metric_name = metric_data['metric']
                score = metric_data['score']

                # Only recommend if score is below threshold
                threshold = 60 if category_name != 'governance' else 70
                if score < threshold and metric_name in recommendation_templates:
                    template = recommendation_templates[metric_name]
                    formatted_value = metric_data['value']
                    if isinstance(formatted_value, (int, float)):
                        formatted_value = f"{formatted_value:,.0f}" if formatted_value > 100 else str(formatted_value)

                    recommendations.append({
                        'category': template['category'],
                        'priority': template['priority'],
                        'metric': template['metric'],
                        'current': f"{formatted_value} {metric_data['unit']}",
                        'score': score,
                        'recommendation': template['recommendation'],
                        'action_items': template['action_items'],
                        'potential_improvement': f"Potential score increase: {100 - score:.0f} points",
                    })

        # Sort by priority then score
        recommendations.sort(key=lambda x: (0 if x['priority'] == 'High' else 1, x['score']))
        return recommendations

    def _create_summary(self, scores: Dict, metrics: Dict, recommendations: List) -> Dict:
        """Create analysis summary."""
        high_priority = len([r for r in recommendations if r['priority'] == 'High'])
        medium_priority = len([r for r in recommendations if r['priority'] == 'Medium'])

        avg_improvement = 0
        if recommendations:
            avg_improvement = sum(100 - r['score'] for r in recommendations) / len(recommendations)

        return {
            'timestamp': datetime.now().isoformat(),
            'total_metrics_extracted': len(metrics),
            'overall_esg_score': scores['overall']['score'],
            'category_scores': {
                'environmental': scores['environmental']['score'],
                'social': scores['social']['score'],
                'governance': scores['governance']['score'],
            },
            'recommendation_count': len(recommendations),
            'high_priority_recommendations': high_priority,
            'medium_priority_recommendations': medium_priority,
            'average_improvement_potential': round(avg_improvement, 1),
            'pipeline_type': 'ml',
            'model_loaded': self.extractor.model_loaded,
        }


def generate_training_data(dataset_dir: str, output_path: str) -> Dict:
    """
    Convenience function to generate training data from the Dataset directory.
    """
    return generate_labeled_dataset(dataset_dir, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test with sample data
        pipeline = ESGMLPipeline(industry='technology')

        test_text = """
        Our total GHG emissions were 50,000 tCO2e.
        Scope 1: 20,000 tCO2e, Scope 2: 15,000 tCO2e, Scope 3: 15,000 tCO2e.
        Renewable energy: 45%. Energy consumption: 75,000 MWh.
        Female representation: 35%. Employee turnover: 12%.
        Board independence: 55%.
        """

        results = pipeline.extractor.extract_from_text(test_text)
        print(f"\nExtracted {len(results)} metrics:")
        for m, info in sorted(results.items()):
            print(f"  {m:25s}: {info['value']} {info['unit']} "
                  f"(conf: {info['confidence']:.2f}, src: {info['source']})")

    elif len(sys.argv) > 1:
        # Analyze a real PDF
        pdf_path = sys.argv[1]
        industry = sys.argv[2] if len(sys.argv) > 2 else 'general'

        pipeline = ESGMLPipeline(industry=industry)
        results = pipeline.analyze_report(pdf_path)

        if results:
            print(f"\n✅ Analysis complete!")
            print(f"  ESG Score: {results['esg_scores']['overall']['score']}/100")
            print(f"  Metrics found: {len(results['extracted_metrics'])}")
            print(f"  Recommendations: {len(results['recommendations'])}")
    else:
        print("Usage:")
        print("  python -m ml_pipeline.pipeline --test         # Run test")
        print("  python -m ml_pipeline.pipeline <pdf_path>     # Analyze PDF")
