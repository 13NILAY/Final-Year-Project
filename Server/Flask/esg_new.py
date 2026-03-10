import re
import pdfplumber
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime
from dataclasses import dataclass
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message="Cannot set gray stroke color")
warnings.filterwarnings('ignore', message="Cannot set gray non-stroke color")

@dataclass
class ESGScoreRanges:
    """Define acceptable ranges for each ESG metric"""
    # Environmental metrics with ideal ranges - ADDED SCOPE 1,2,3 AND CO2
    ENVIRONMENTAL = {
        'ghg_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.15, 'threshold': 100000},
        'scope1_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 50000},
        'scope2_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 50000},
        'scope3_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 100000},
        'co2_emissions': {'unit': 'tCO2e', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 80000},
        'energy_consumption': {'unit': 'MWh', 'ideal': 'decreasing', 'weight': 0.10, 'threshold': 50000},
        'renewable_energy': {'unit': '%', 'ideal': (30, 100), 'weight': 0.10},
        'water_withdrawal': {'unit': 'm³', 'ideal': 'decreasing', 'weight': 0.08, 'threshold': 1000000},
        'waste_recycled': {'unit': '%', 'ideal': (60, 100), 'weight': 0.07},
        'hazardous_waste': {'unit': 'tons', 'ideal': 'decreasing', 'weight': 0.05, 'threshold': 1000}
    }
    
    # Social metrics with ideal ranges
    SOCIAL = {
        'employee_turnover': {'unit': '%', 'ideal': (0, 15), 'weight': 0.15},
        'female_representation': {'unit': '%', 'ideal': (40, 100), 'weight': 0.20},
        'training_hours': {'unit': 'hours', 'ideal': (20, 100), 'weight': 0.15},
        'lost_time_injury': {'unit': 'rate', 'ideal': (0, 2), 'weight': 0.25},
        'employee_satisfaction': {'unit': '%', 'ideal': (70, 100), 'weight': 0.15},
        'community_investment': {'unit': 'USD', 'ideal': 'increasing', 'weight': 0.10, 'threshold': 1000000}
    }
    
    # Governance metrics with ideal ranges
    GOVERNANCE = {
        'board_independence': {'unit': '%', 'ideal': (50, 100), 'weight': 0.25},
        'female_directors': {'unit': '%', 'ideal': (30, 100), 'weight': 0.20},
        'ceo_pay_ratio': {'unit': 'ratio', 'ideal': (1, 50), 'weight': 0.25},
        'ethics_training': {'unit': '%', 'ideal': (90, 100), 'weight': 0.15},
        'whistleblower_cases': {'unit': 'count', 'ideal': (0, 5), 'weight': 0.15}
    }

@dataclass
class TestCase:
    """Individual test case for validation"""
    text: str
    expected_metrics: Dict[str, float]
    expected_units: Dict[str, str]
    description: str
    category: str = "general"

class ESGAccuracyValidator:
    """
    Comprehensive accuracy testing framework for ESG metric extraction
    """
    
    def __init__(self, analyzer: 'ESGReportAnalyzer'):
        self.analyzer = analyzer
        self.test_cases = []
        self.results = []
        
    def create_validation_suite(self) -> List[TestCase]:
        """
        Create comprehensive test cases for validation - ADDED SCOPE 1,2,3 AND CO2 TEST CASES
        """
        test_cases = []
        
        # Test Case 1: Clear metric with unit
        test_cases.append(TestCase(
            text="Our total GHG emissions were 125,000 tCO2e last year, representing a 5% reduction.",
            expected_metrics={'ghg_emissions': 125000},
            expected_units={'ghg_emissions': 'tCO2e'},
            description="Clear GHG emissions with unit",
            category="environmental"
        ))
        
        # NEW: Scope 1 emissions test
        test_cases.append(TestCase(
            text="Scope 1 emissions: 45,000 tCO2e from direct operations.",
            expected_metrics={'scope1_emissions': 45000},
            expected_units={'scope1_emissions': 'tCO2e'},
            description="Scope 1 emissions",
            category="environmental"
        ))
        
        # NEW: Scope 2 emissions test
        test_cases.append(TestCase(
            text="Scope 2 emissions were 35,200 tCO2e from purchased electricity.",
            expected_metrics={'scope2_emissions': 35200},
            expected_units={'scope2_emissions': 'tCO2e'},
            description="Scope 2 emissions",
            category="environmental"
        ))
        
        # NEW: Scope 3 emissions test
        test_cases.append(TestCase(
            text="Scope 3 emissions totaled 280,000 tCO2e across our value chain.",
            expected_metrics={'scope3_emissions': 280000},
            expected_units={'scope3_emissions': 'tCO2e'},
            description="Scope 3 emissions",
            category="environmental"
        ))
        
        # NEW: CO2 emissions test
        test_cases.append(TestCase(
            text="CO2 emissions decreased to 95,000 tonnes this year.",
            expected_metrics={'co2_emissions': 95000},
            expected_units={'co2_emissions': 'tons'},
            description="CO2 emissions",
            category="environmental"
        ))
        
        # Test Case 2: Percentage metric
        test_cases.append(TestCase(
            text="Renewable energy accounted for 35% of our total energy consumption.",
            expected_metrics={'renewable_energy': 35},
            expected_units={'renewable_energy': '%'},
            description="Percentage renewable energy",
            category="environmental"
        ))
        
        # Test Case 3: Multiple metrics in same text
        test_cases.append(TestCase(
            text="Female representation in our workforce increased to 42.5%. Employee turnover was 12.3%.",
            expected_metrics={'female_representation': 42.5, 'employee_turnover': 12.3},
            expected_units={'female_representation': '%', 'employee_turnover': '%'},
            description="Multiple social metrics",
            category="social"
        ))
        
        # Test Case 4: Large numbers with commas
        test_cases.append(TestCase(
            text="Water withdrawal: 2,500,000 m³",
            expected_metrics={'water_withdrawal': 2500000},
            expected_units={'water_withdrawal': 'm³'},
            description="Large number with commas",
            category="environmental"
        ))
        
        # Test Case 5: Different phrasing
        test_cases.append(TestCase(
            text="Board independence stands at 60 percent, with 40% female directors.",
            expected_metrics={'board_independence': 60, 'female_directors': 40},
            expected_units={'board_independence': '%', 'female_directors': '%'},
            description="Different phrasing for percentages",
            category="governance"
        ))
        
        # Test Case 6: No metrics (negative test)
        test_cases.append(TestCase(
            text="Our company is committed to sustainability and ethical practices.",
            expected_metrics={},
            expected_units={},
            description="Text with no metrics",
            category="negative"
        ))
        
        # Test Case 7: Metric with context
        test_cases.append(TestCase(
            text="CEO pay ratio: 120:1 (CEO to median employee)",
            expected_metrics={'ceo_pay_ratio': 120},
            expected_units={'ceo_pay_ratio': 'ratio'},
            description="CEO pay ratio with context",
            category="governance"
        ))
        
        # Test Case 8: Training hours
        test_cases.append(TestCase(
            text="Average training hours per employee: 28.5 hours annually",
            expected_metrics={'training_hours': 28.5},
            expected_units={'training_hours': 'hours'},
            description="Training hours with unit",
            category="social"
        ))
        
        # Test Case 9: Waste recycled
        test_cases.append(TestCase(
            text="Waste recycled reached 78% in the reporting period",
            expected_metrics={'waste_recycled': 78},
            expected_units={'waste_recycled': '%'},
            description="Waste recycling percentage",
            category="environmental"
        ))
        
        # Test Case 10: Community investment
        test_cases.append(TestCase(
            text="Community investment totaled $2.5 million USD",
            expected_metrics={'community_investment': 2500000},
            expected_units={'community_investment': 'USD'},
            description="Community investment in millions",
            category="social"
        ))
        
        # Test Case 11: Hazardous waste
        test_cases.append(TestCase(
            text="Hazardous waste generation: 850 tonnes",
            expected_metrics={'hazardous_waste': 850},
            expected_units={'hazardous_waste': 'tons'},
            description="Hazardous waste in tonnes",
            category="environmental"
        ))
        
        # Test Case 12: Energy consumption
        test_cases.append(TestCase(
            text="Total energy consumption was 45,200 MWh",
            expected_metrics={'energy_consumption': 45200},
            expected_units={'energy_consumption': 'MWh'},
            description="Energy consumption",
            category="environmental"
        ))
        
        # NEW: Multiple scope emissions test
        test_cases.append(TestCase(
            text="Our GHG emissions: Scope 1: 12,500 tCO2e, Scope 2: 28,300 tCO2e, Scope 3: 156,000 tCO2e.",
            expected_metrics={'scope1_emissions': 12500, 'scope2_emissions': 28300, 'scope3_emissions': 156000},
            expected_units={'scope1_emissions': 'tCO2e', 'scope2_emissions': 'tCO2e', 'scope3_emissions': 'tCO2e'},
            description="Multiple scope emissions",
            category="environmental"
        ))
        
        # NEW: Combined GHG and scope test
        test_cases.append(TestCase(
            text="Total GHG emissions: 450,000 tCO2e (Scope 1: 120,000, Scope 2: 80,000, Scope 3: 250,000)",
            expected_metrics={'ghg_emissions': 450000, 'scope1_emissions': 120000, 'scope2_emissions': 80000, 'scope3_emissions': 250000},
            expected_units={'ghg_emissions': 'tCO2e', 'scope1_emissions': 'tCO2e', 'scope2_emissions': 'tCO2e', 'scope3_emissions': 'tCO2e'},
            description="Combined GHG and scope emissions",
            category="environmental"
        ))
        
        return test_cases
    
    def run_validation(self, test_cases: List[TestCase] = None) -> Dict:
        """
        Run validation on test cases and calculate accuracy metrics
        """
        if test_cases is None:
            test_cases = self.create_validation_suite()
        
        results = {
            'total_tests': len(test_cases),
            'metrics_found': 0,
            'metrics_correct': 0,
            'unit_correct': 0,
            'detailed_results': []
        }
        
        # Metrics for confusion matrix
        all_expected = []
        all_predicted = []
        all_metric_names = []
        
        for i, test_case in enumerate(test_cases):
            # Extract metrics using analyzer
            extracted = self.analyzer.extract_specific_metrics(test_case.text)
            
            # Compare with expected
            case_result = self._evaluate_test_case(test_case, extracted)
            
            # Update totals
            results['metrics_found'] += case_result['metrics_extracted']
            results['metrics_correct'] += case_result['metrics_correct']
            results['unit_correct'] += case_result['units_correct']
            
            # For confusion matrix
            for metric_name, expected_value in test_case.expected_metrics.items():
                predicted_value = extracted.get(metric_name, {}).get('value')
                
                # Convert to binary classification (found/not found)
                all_expected.append(1)  # Expected to find
                all_predicted.append(1 if predicted_value is not None else 0)
                all_metric_names.append(metric_name)
            
            results['detailed_results'].append(case_result)
        
        # Calculate accuracy metrics
        results['extraction_accuracy'] = self._calculate_accuracy(results)
        results['precision_recall_f1'] = self._calculate_precision_recall_f1(all_expected, all_predicted)
        results['category_accuracy'] = self._calculate_category_accuracy(results['detailed_results'])
        
        return results
    
    def _evaluate_test_case(self, test_case: TestCase, extracted: Dict) -> Dict:
        """
        Evaluate a single test case
        """
        result = {
            'test_description': test_case.description,
            'category': test_case.category,
            'expected_metrics': len(test_case.expected_metrics),
            'metrics_extracted': 0,
            'metrics_correct': 0,
            'units_correct': 0,
            'details': []
        }
        
        for metric_name, expected_value in test_case.expected_metrics.items():
            if metric_name in extracted:
                extracted_value = extracted[metric_name]['value']
                extracted_unit = extracted[metric_name].get('unit', '')
                expected_unit = test_case.expected_units.get(metric_name, '')
                
                # Check if value is correct (within tolerance)
                tolerance = self._get_tolerance(metric_name)
                is_value_correct = self._is_within_tolerance(extracted_value, expected_value, tolerance)
                is_unit_correct = (extracted_unit == expected_unit) or self._are_units_equivalent(extracted_unit, expected_unit)
                
                result['metrics_extracted'] += 1
                if is_value_correct:
                    result['metrics_correct'] += 1
                if is_unit_correct:
                    result['units_correct'] += 1
                
                result['details'].append({
                    'metric': metric_name,
                    'expected_value': expected_value,
                    'extracted_value': extracted_value,
                    'expected_unit': expected_unit,
                    'extracted_unit': extracted_unit,
                    'value_correct': is_value_correct,
                    'unit_correct': is_unit_correct,
                    'tolerance_used': tolerance
                })
        
        return result
    
    def _get_tolerance(self, metric_name: str) -> float:
        """Get tolerance for different metric types - UPDATED WITH NEW METRICS"""
        tolerances = {
            # Environmental - including new metrics
            'ghg_emissions': 0.02,
            'scope1_emissions': 0.02,
            'scope2_emissions': 0.02,
            'scope3_emissions': 0.03,  # Slightly looser for Scope 3
            'co2_emissions': 0.02,
            'energy_consumption': 0.02,
            'renewable_energy': 0.01,
            'water_withdrawal': 0.02,
            'waste_recycled': 0.02,
            'hazardous_waste': 0.02,
            
            # Social
            'employee_turnover': 0.02,
            'female_representation': 0.01,
            'training_hours': 0.05,
            'lost_time_injury': 0.02,
            'employee_satisfaction': 0.02,
            'community_investment': 0.03,
            
            # Governance
            'board_independence': 0.02,
            'female_directors': 0.01,
            'ceo_pay_ratio': 0.03,
            'ethics_training': 0.02,
            'whistleblower_cases': 0.0
        }
        return tolerances.get(metric_name, 0.05)
    
    def _is_within_tolerance(self, extracted: float, expected: float, tolerance: float) -> bool:
        """Check if extracted value is within tolerance of expected"""
        if extracted is None or expected is None:
            return False
        
        try:
            relative_error = abs(extracted - expected) / expected
            return relative_error <= tolerance
        except (TypeError, ZeroDivisionError):
            # For non-numeric values, require exact match
            return str(extracted) == str(expected)
    
    def _are_units_equivalent(self, unit1: str, unit2: str) -> bool:
        """Check if units are equivalent (e.g., tons = tonnes)"""
        unit_equivalents = {
            'tons': ['tonnes', 't', 'ton'],
            'tonnes': ['tons', 't', 'ton'],
            '%': ['percent', 'percentage', 'pct'],
            'm³': ['cubic meters', 'cubic metres', 'm3'],
            'MWh': ['megawatt-hour', 'megawatt hours', 'mwh'],
            'USD': ['$', 'us dollars', 'dollars'],
            'hours': ['hrs', 'hour', 'hr'],
            'ratio': [':1', 'times', 'x'],
            'tCO2e': ['tco2e', 'tons co2', 'tonnes co2', 't co2', 'mtco2e', 'ktco2e']
        }
        
        unit1_lower = unit1.lower()
        unit2_lower = unit2.lower()
        
        if unit1_lower == unit2_lower:
            return True
        
        # Check if units are in each other's equivalence lists
        for key, equivalents in unit_equivalents.items():
            if unit1_lower == key.lower() and unit2_lower in equivalents:
                return True
            if unit2_lower == key.lower() and unit1_lower in equivalents:
                return True
        
        return False
    
    def _calculate_accuracy(self, results: Dict) -> Dict:
        """Calculate various accuracy metrics"""
        if results['metrics_found'] == 0:
            return {
                'extraction_rate': 0,
                'value_accuracy': 0,
                'unit_accuracy': 0,
                'overall_accuracy': 0
            }
        
        extraction_rate = results['metrics_found'] / sum(r['expected_metrics'] for r in results['detailed_results']) * 100
        value_accuracy = results['metrics_correct'] / results['metrics_found'] * 100 if results['metrics_found'] > 0 else 0
        unit_accuracy = results['unit_correct'] / results['metrics_found'] * 100 if results['metrics_found'] > 0 else 0
        
        # Overall accuracy (weighted average)
        overall_accuracy = (extraction_rate * 0.4 + value_accuracy * 0.4 + unit_accuracy * 0.2)
        
        return {
            'extraction_rate': round(extraction_rate, 2),
            'value_accuracy': round(value_accuracy, 2),
            'unit_accuracy': round(unit_accuracy, 2),
            'overall_accuracy': round(overall_accuracy, 2)
        }
    
    def _calculate_precision_recall_f1(self, expected: List[int], predicted: List[int]) -> Dict:
        """Calculate precision, recall, and F1 score"""
        if len(expected) == 0 or len(predicted) == 0:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        
        precision = precision_score(expected, predicted, zero_division=0)
        recall = recall_score(expected, predicted, zero_division=0)
        f1 = f1_score(expected, predicted, zero_division=0)
        
        return {
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2)
        }
    
    def _calculate_category_accuracy(self, detailed_results: List[Dict]) -> Dict:
        """Calculate accuracy by metric category"""
        categories = {}
        
        for result in detailed_results:
            category = result['category']
            if category not in categories:
                categories[category] = {
                    'total': 0,
                    'found': 0,
                    'correct': 0
                }
            
            categories[category]['total'] += result['expected_metrics']
            categories[category]['found'] += result['metrics_extracted']
            categories[category]['correct'] += result['metrics_correct']
        
        # Calculate accuracy per category
        category_accuracy = {}
        for category, stats in categories.items():
            if stats['total'] > 0:
                extraction_rate = stats['found'] / stats['total'] * 100
                if stats['found'] > 0:
                    accuracy = stats['correct'] / stats['found'] * 100
                else:
                    accuracy = 0
                category_accuracy[category] = {
                    'extraction_rate': round(extraction_rate, 2),
                    'accuracy': round(accuracy, 2)
                }
        
        return category_accuracy
    
    def generate_validation_report(self, results: Dict, output_path: str = None):
        """
        Generate detailed validation report
        """
        print("\n" + "="*80)
        print("ESG METRIC EXTRACTION VALIDATION REPORT")
        print("="*80)
        
        # Summary metrics
        accuracy = results['extraction_accuracy']
        prf = results['precision_recall_f1']
        
        print(f"\n📊 OVERALL ACCURACY: {accuracy['overall_accuracy']}%")
        print("-"*60)
        print(f"Extraction Rate:      {accuracy['extraction_rate']}%")
        print(f"Value Accuracy:       {accuracy['value_accuracy']}%")
        print(f"Unit Accuracy:        {accuracy['unit_accuracy']}%")
        
        print(f"\n🤖 CLASSIFICATION METRICS:")
        print(f"Precision:            {prf['precision']}%")
        print(f"Recall:               {prf['recall']}%")
        print(f"F1 Score:             {prf['f1']}%")
        
        print(f"\n📈 CATEGORY PERFORMANCE:")
        for category, stats in results['category_accuracy'].items():
            print(f"{category.capitalize():<20} Extraction: {stats['extraction_rate']:>6}% | Accuracy: {stats['accuracy']:>6}%")
        
        # Pass/fail based on target
        target_accuracy = 80
        if accuracy['overall_accuracy'] >= target_accuracy:
            print(f"\n✅ PASS: Overall accuracy meets target ({target_accuracy}%)")
        else:
            print(f"\n❌ FAIL: Overall accuracy below target ({target_accuracy}%)")
        
        # Detailed results
        print(f"\n🔍 DETAILED RESULTS ({len(results['detailed_results'])} test cases):")
        print("-"*60)
        
        for i, case_result in enumerate(results['detailed_results'], 1):
            print(f"\nTest {i}: {case_result['test_description']}")
            print(f"Category: {case_result['category']}")
            print(f"Expected: {case_result['expected_metrics']} metrics | Found: {case_result['metrics_extracted']} | Correct: {case_result['metrics_correct']}")
            
            for detail in case_result['details']:
                status = "✓" if detail['value_correct'] else "✗"
                unit_status = "✓" if detail['unit_correct'] else "✗"
                print(f"  {status}{unit_status} {detail['metric']}: Expected {detail['expected_value']} {detail['expected_unit']}, "
                      f"Got {detail['extracted_value']} {detail['extracted_unit']}")
        
        # Generate visualization if matplotlib is available
        try:
            self._create_visualization(results, output_path)
        except ImportError:
            print("\n⚠ Matplotlib not available for visualizations")
        
        # Save detailed report to file
        if output_path:
            self._save_detailed_report(results, output_path)
    
    def _create_visualization(self, results: Dict, output_path: str = None):
        """Create accuracy visualization charts"""
        plt.figure(figsize=(15, 10))
        
        # Chart 1: Accuracy breakdown
        plt.subplot(2, 2, 1)
        accuracy_data = results['extraction_accuracy']
        labels = ['Extraction Rate', 'Value Accuracy', 'Unit Accuracy', 'Overall']
        values = [accuracy_data['extraction_rate'], accuracy_data['value_accuracy'], 
                 accuracy_data['unit_accuracy'], accuracy_data['overall_accuracy']]
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        bars = plt.bar(labels, values, color=colors)
        plt.title('Accuracy Breakdown')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Chart 2: Category performance
        plt.subplot(2, 2, 2)
        category_data = results['category_accuracy']
        categories = list(category_data.keys())
        extraction_rates = [category_data[c]['extraction_rate'] for c in categories]
        accuracies = [category_data[c]['accuracy'] for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, extraction_rates, width, label='Extraction Rate', color='skyblue')
        plt.bar(x + width/2, accuracies, width, label='Accuracy', color='lightgreen')
        
        plt.title('Performance by Category')
        plt.ylabel('Percentage (%)')
        plt.xticks(x, [c.capitalize() for c in categories])
        plt.legend()
        plt.ylim(0, 100)
        
        # Chart 3: Precision-Recall-F1
        plt.subplot(2, 2, 3)
        prf_data = results['precision_recall_f1']
        metrics = ['Precision', 'Recall', 'F1 Score']
        values_prf = [prf_data['precision'], prf_data['recall'], prf_data['f1']]
        
        plt.bar(metrics, values_prf, color=['orange', 'purple', 'brown'])
        plt.title('Classification Metrics')
        plt.ylabel('Score (%)')
        plt.ylim(0, 100)
        
        for i, value in enumerate(values_prf):
            plt.text(i, value + 1, f'{value:.1f}%', ha='center')
        
        # Chart 4: Test case results
        plt.subplot(2, 2, 4)
        test_results = results['detailed_results']
        test_numbers = range(1, len(test_results) + 1)
        extraction_performance = []
        
        for result in test_results:
            if result['expected_metrics'] > 0:
                perf = result['metrics_correct'] / result['expected_metrics'] * 100
            else:
                perf = 100 if result['metrics_extracted'] == 0 else 0
            extraction_performance.append(perf)
        
        plt.bar(test_numbers, extraction_performance, color='teal')
        plt.title('Individual Test Case Performance')
        plt.xlabel('Test Case Number')
        plt.ylabel('Performance (%)')
        plt.ylim(0, 100)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}_visualization.png", dpi=300, bbox_inches='tight')
            print(f"\n📊 Visualization saved to: {output_path}_visualization.png")
        plt.show()
    
    def _save_detailed_report(self, results: Dict, output_path: str):
        """Save detailed validation report to JSON"""
        report_data = {
            'validation_date': datetime.now().isoformat(),
            'analyzer_config': {
                'industry': self.analyzer.industry,
                'patterns_used': list(self.analyzer.metric_patterns.keys())
            },
            'accuracy_metrics': results['extraction_accuracy'],
            'classification_metrics': results['precision_recall_f1'],
            'category_performance': results['category_accuracy'],
            'test_summary': {
                'total_tests': results['total_tests'],
                'total_expected_metrics': sum(r['expected_metrics'] for r in results['detailed_results']),
                'metrics_found': results['metrics_found'],
                'metrics_correct': results['metrics_correct'],
                'units_correct': results['unit_correct']
            },
            'detailed_test_results': results['detailed_results']
        }
        
        with open(f"{output_path}_detailed.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {output_path}_detailed.json")
        
        # Also save as CSV for analysis
        self._save_results_to_csv(results, f"{output_path}_results.csv")
    
    def _save_results_to_csv(self, results: Dict, csv_path: str):
        """Save results to CSV for further analysis"""
        rows = []
        
        for i, case_result in enumerate(results['detailed_results'], 1):
            for detail in case_result['details']:
                rows.append({
                    'test_number': i,
                    'test_description': case_result['test_description'],
                    'category': case_result['category'],
                    'metric': detail['metric'],
                    'expected_value': detail['expected_value'],
                    'extracted_value': detail['extracted_value'],
                    'expected_unit': detail['expected_unit'],
                    'extracted_unit': detail['extracted_unit'],
                    'value_correct': detail['value_correct'],
                    'unit_correct': detail['unit_correct'],
                    'tolerance_used': detail['tolerance_used']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"📊 Results CSV saved to: {csv_path}")

class ESGReportAnalyzer:
    """
    Advanced ESG Report Analyzer with proper scoring and recommendations
    """
    
    def __init__(self, industry_norm: str = 'general'):
        """
        Initialize analyzer with industry-specific norms
        """
        self.industry_norms = {
            'general': ESGScoreRanges(),
            'mining': self._get_mining_norms(),
            'manufacturing': self._get_manufacturing_norms(),
            'technology': self._get_technology_norms(),
            'finance': self._get_finance_norms()
        }
        
        self.current_norms = self.industry_norms.get(industry_norm, ESGScoreRanges())
        self.industry = industry_norm
        
        # Patterns for extracting specific metrics with values - UPDATED WITH SCOPE AND CO2 PATTERNS
        self.metric_patterns = self._initialize_patterns()
        
        # Category mapping for quick lookup
        self._category_map = self._build_category_map()
    
    def _build_category_map(self):
        """Build mapping of metrics to categories - UPDATED WITH NEW METRICS"""
        category_map = {}
        
        # Map environmental metrics including new ones
        env_metrics = [
            'ghg_emissions', 'scope1_emissions', 'scope2_emissions', 'scope3_emissions',
            'co2_emissions', 'energy_consumption', 'renewable_energy', 
            'water_withdrawal', 'waste_recycled', 'hazardous_waste'
        ]
        for metric in env_metrics:
            category_map[metric] = 'environmental'
        
        # Map social metrics
        for metric in self.current_norms.SOCIAL.keys():
            category_map[metric] = 'social'
        
        # Map governance metrics
        for metric in self.current_norms.GOVERNANCE.keys():
            category_map[metric] = 'governance'
        
        return category_map
    
    def _initialize_patterns(self):
        """Initialize regex patterns for metric extraction - ADDED SCOPE AND CO2 PATTERNS"""
        return {
            # Environmental patterns - ENHANCED WITH SCOPE 1,2,3 AND CO2
            'ghg_emissions': [
                r'(?:total\s+)?(?:greenhouse\s+gas|ghg|carbon)\s+emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?|mt)?',
                r'([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)\s+(?:of\s+)?(?:greenhouse\s+gas|ghg|carbon)\s+emissions',
                r'emissions.*?total.*?([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'ghg.*?inventory.*?([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)'
            ],
            'scope1_emissions': [
                r'scope\s*1[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?|mt)?',
                r'direct\s+emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'scope\s*one[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r's1[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'scope\s*1.*?([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)'
            ],
            'scope2_emissions': [
                r'scope\s*2[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?|mt)?',
                r'indirect\s+emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'scope\s*two[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r's2[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'scope\s*2.*?([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)'
            ],
            'scope3_emissions': [
                r'scope\s*3[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?|mt)?',
                r'value\s*chain\s+emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'scope\s*three[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r's3[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'other\s+indirect\s+emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'scope\s*3.*?([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)'
            ],
            'co2_emissions': [
                r'co2\s+emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?|mt)?',
                r'carbon\s+dioxide\s+emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'co2e?\s*:?\s*([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)',
                r'co2.*?emissions.*?([\d,]+\.?\d*)\s*(?:tco2e|tons?|tonnes?)'
            ],
            'energy_consumption': [
                r'(?:total\s+)?energy\s+(?:consumption|use|usage)[:\s]*([\d,]+\.?\d*)\s*(?:mwh|gwh|kwh|j)',
                r'energy\s+use[:\s]*([\d,]+\.?\d*)\s*(?:mwh|gwh)',
                r'consumed.*?([\d,]+\.?\d*)\s*(?:mwh|gwh).*?energy'
            ],
            'renewable_energy': [
                r'renewable\s+energy[:\s]*([\d,]*\.?\d*)\s*%',
                r'([\d,]*\.?\d*)\s*%\s+(?:of\s+)?energy\s+from\s+renewable',
                r'renewables.*?([\d,]*\.?\d*)\s*%',
                r'green\s+energy.*?([\d,]*\.?\d*)\s*%'
            ],
            'water_withdrawal': [
                r'water\s+(?:withdrawal|consumption|usage)[:\s]*([\d,]+\.?\d*)\s*(?:m³|cubic meters|liters|gallons|kl|ml)',
                r'([\d,]+\.?\d*)\s*(?:m³|million liters)\s+of\s+water',
                r'water.*?([\d,]+\.?\d*)\s*(?:m³|ml)'
            ],
            'waste_recycled': [
                r'waste\s+recycled[:\s]*([\d,]*\.?\d*)\s*%',
                r'recycling\s+rate[:\s]*([\d,]*\.?\d*)\s*%',
                r'([\d,]*\.?\d*)\s*%\s+(?:of\s+)?waste.*?recycled'
            ],
            'hazardous_waste': [
                r'hazardous\s+waste[:\s]*([\d,]+\.?\d*)\s*(?:tons|tonnes)',
                r'([\d,]+\.?\d*)\s*(?:tons?|tonnes?).*?hazardous\s+waste'
            ],
            
            # Social patterns
            'employee_turnover': [
                r'employee\s+turnover\s+rate[:\s]*([\d,]*\.?\d*)\s*%',
                r'turnover\s+rate[:\s]*([\d,]*\.?\d*)\s*%',
                r'turnover.*?([\d,]*\.?\d*)\s*%'
            ],
            'female_representation': [
                r'female\s+(?:representation|employees|workforce)[:\s]*([\d,]*\.?\d*)\s*%',
                r'women\s+(?:in\s+)?workforce[:\s]*([\d,]*\.?\d*)\s*%',
                r'gender.*?diversity.*?([\d,]*\.?\d*)\s*%'
            ],
            'training_hours': [
                r'average\s+training\s+hours[:\s]*([\d,]+\.?\d*)\s*hours',
                r'training\s+hours\s+per\s+employee[:\s]*([\d,]+\.?\d*)',
                r'([\d,]+\.?\d*)\s*hours.*?training'
            ],
            'lost_time_injury': [
                r'lost\s+time\s+injury\s+rate[:\s]*([\d,]*\.?\d*)',
                r'ltir[:\s]*([\d,]*\.?\d*)',
                r'injury.*?rate.*?([\d,]*\.?\d*)'
            ],
            'employee_satisfaction': [
                r'employee\s+satisfaction[:\s]*([\d,]*\.?\d*)\s*%',
                r'engagement\s+score[:\s]*([\d,]*\.?\d*)\s*%',
                r'satisfaction.*?([\d,]*\.?\d*)\s*%'
            ],
            'community_investment': [
                r'community\s+investment[:\s]*([\d,]+\.?\d*)\s*(?:usd|\$|million|billion)',
                r'charitable.*?contributions.*?([\d,]+\.?\d*)\s*(?:usd|\$)',
                r'philanthrop.*?([\d,]+\.?\d*)\s*(?:usd|\$)'
            ],
            
            # Governance patterns
            'board_independence': [
                r'independent\s+directors[:\s]*([\d,]*\.?\d*)\s*%',
                r'board\s+independence[:\s]*([\d,]*\.?\d*)\s*%',
                r'([\d,]*\.?\d*)\s*%.*?independent.*?directors'
            ],
            'female_directors': [
                r'female\s+board\s+members[:\s]*([\d,]*\.?\d*)\s*%',
                r'women\s+on\s+board[:\s]*([\d,]*\.?\d*)\s*%',
                r'board.*?diversity.*?([\d,]*\.?\d*)\s*%'
            ],
            'ceo_pay_ratio': [
                r'ceo[-]?\s*pay\s+ratio[:\s]*([\d,]+\.?\d*):?1?',
                r'ceo[\s\-]to[\s\-]worker\s+pay[:\s]*([\d,]+\.?\d*)',
                r'pay\s+ratio.*?([\d,]+\.?\d*)'
            ],
            'ethics_training': [
                r'ethics\s+training\s+completion[:\s]*([\d,]*\.?\d*)\s*%',
                r'anti[\s\-]?corruption\s+training[:\s]*([\d,]*\.?\d*)\s*%',
                r'([\d,]*\.?\d*)\s*%.*?ethics.*?training'
            ],
            'whistleblower_cases': [
                r'whistleblower\s+cases[:\s]*([\d,]+\.?\d*)',
                r'([\d,]+\.?\d*)\s*whistleblower\s+cases',
                r'reported.*?cases.*?([\d,]+\.?\d*)'
            ]
        }
    
    def _get_mining_norms(self):
        """Industry-specific norms for mining"""
        norms = ESGScoreRanges()
        norms.ENVIRONMENTAL['water_withdrawal']['weight'] = 0.20
        norms.ENVIRONMENTAL['hazardous_waste']['weight'] = 0.15
        norms.ENVIRONMENTAL['scope1_emissions']['weight'] = 0.15
        norms.ENVIRONMENTAL['scope2_emissions']['weight'] = 0.10
        norms.SOCIAL['lost_time_injury']['weight'] = 0.35
        norms.SOCIAL['training_hours']['ideal'] = (40, 100)
        return norms
    
    def _get_technology_norms(self):
        """Industry-specific norms for technology"""
        norms = ESGScoreRanges()
        norms.ENVIRONMENTAL['energy_consumption']['weight'] = 0.25
        norms.ENVIRONMENTAL['scope2_emissions']['weight'] = 0.15
        norms.SOCIAL['female_representation']['weight'] = 0.30
        norms.SOCIAL['female_representation']['ideal'] = (35, 100)
        norms.GOVERNANCE['ethics_training']['weight'] = 0.20
        return norms
    
    def _get_manufacturing_norms(self):
        """Industry-specific norms for manufacturing"""
        norms = ESGScoreRanges()
        norms.ENVIRONMENTAL['ghg_emissions']['weight'] = 0.25
        norms.ENVIRONMENTAL['scope1_emissions']['weight'] = 0.15
        norms.ENVIRONMENTAL['waste_recycled']['weight'] = 0.15
        norms.SOCIAL['employee_turnover']['weight'] = 0.20
        return norms
    
    def _get_finance_norms(self):
        """Industry-specific norms for finance"""
        norms = ESGScoreRanges()
        norms.ENVIRONMENTAL['scope3_emissions']['weight'] = 0.25
        norms.SOCIAL['community_investment']['weight'] = 0.20
        norms.GOVERNANCE['board_independence']['weight'] = 0.30
        norms.GOVERNANCE['ceo_pay_ratio']['weight'] = 0.30
        return norms
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with improved accuracy and error handling"""
        full_text = ""
        pages_extracted = 0
        pages_failed = 0
        
        try:
            # Temporarily suppress pdfplumber warnings
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message="Cannot set gray stroke color")
                warnings.filterwarnings('ignore', message="Cannot set gray non-stroke color")
                
                with pdfplumber.open(pdf_path) as pdf:
                    total_pages = len(pdf.pages)
                    print(f"   Processing {total_pages} pages...")
                    
                    for i, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text:
                                full_text += text + "\n\n"
                                pages_extracted += 1
                        except Exception as page_error:
                            # Skip problematic pages but continue
                            pages_failed += 1
                            if pages_failed <= 3:  # Only show first 3 errors
                                print(f"   Warning: Could not extract page {i+1}: {str(page_error)[:80]}")
                            continue
                    
                    print(f"   ✓ Successfully extracted {pages_extracted}/{total_pages} pages")
                    if pages_failed > 0:
                        print(f"   ⚠ Failed to extract {pages_failed} pages (skipped)")
                            
            if not full_text:
                raise ValueError("No text could be extracted from PDF. The file may be image-based, corrupted, or require OCR.")
                    
        except Exception as e:
            print(f"   ❌ Error extracting from PDF: {str(e)[:200]}")
            print(f"   Tip: If this is a scanned PDF, you may need OCR preprocessing.")
            return ""
                
        return full_text
    
    def extract_specific_metrics(self, text: str) -> Dict[str, Dict]:
        """
        Extract specific ESG metrics with their values and context
        """
        if not text:
            print("   ⚠ No text available for metric extraction")
            return {}
            
        extracted = {}
        text_lower = text.lower()
        
        for metric_name, patterns in self.metric_patterns.items():
            metric_values = []
            
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    try:
                        value_str = match.group(1).replace(',', '').strip()
                        
                        # Try to convert to appropriate numeric type
                        try:
                            if '%' in match.group(0) or 'rate' in metric_name:
                                numeric_value = float(value_str)
                            else:
                                # Check if it's a large number that might have "million" or "billion"
                                match_text = match.group(0).lower()
                                if 'million' in match_text:
                                    numeric_value = float(value_str) * 1000000
                                elif 'billion' in match_text:
                                    numeric_value = float(value_str) * 1000000000
                                elif 'thousand' in match_text:
                                    numeric_value = float(value_str) * 1000
                                else:
                                    numeric_value = float(value_str)
                                
                                # Convert to int if it's a whole number
                                if numeric_value.is_integer():
                                    numeric_value = int(numeric_value)
                        except (ValueError, AttributeError):
                            numeric_value = value_str  # Keep as string if conversion fails
                        
                        # Get context
                        start = max(0, match.start() - 150)
                        end = min(len(text_lower), match.end() + 150)
                        context = text_lower[start:end].replace('\n', ' ').strip()
                        
                        metric_values.append({
                            'value': numeric_value,
                            'raw_match': match.group(0),
                            'context': context,
                            'unit': self._get_unit_from_match(match.group(0), metric_name),
                            'confidence': self._calculate_confidence(match.group(0), context)
                        })
                    except (IndexError, AttributeError):
                        continue
            
            if metric_values:
                # Select the best value
                best_value = self._select_best_value(metric_values, metric_name)
                if best_value:
                    extracted[metric_name] = best_value
        
        return extracted
    
    def _calculate_confidence(self, match_text: str, context: str) -> float:
        """Calculate confidence score for extracted value - IMPROVED"""
        confidence = 0.6  # Base confidence (increased from 0.5)
        
        # Increase confidence for specific patterns
        if ':' in match_text or '=' in match_text:
            confidence += 0.1
        
        if '%' in match_text:
            confidence += 0.1
        
        # Check for unit in match
        units = ['tco2e', 'mwh', '%', 'tons', 'hours', 'rate', 'usd', 'ratio', 'm³', 'scope', 'co2']
        unit_count = sum(1 for unit in units if unit in match_text.lower())
        confidence += unit_count * 0.05
        
        # Check if context has ESG-related terms
        esg_terms = ['report', 'metric', 'performance', 'target', 'goal', 'reduction', 
                     'emissions', 'consumption', 'waste', 'water', 'energy', 'diversity',
                     'board', 'training', 'safety', 'scope', 'ghg', 'co2']
        term_count = sum(1 for term in esg_terms if term in context)
        confidence += min(term_count * 0.02, 0.2)  # Max 0.2 boost
        
        # Penalize uncertainty indicators
        uncertainty_words = ['approximately', 'about', 'around', 'estimated', 'approx', 'nearly']
        if any(word in context for word in uncertainty_words):
            confidence -= 0.1
        
        return min(confidence, 1.0)
    
    def _get_unit_from_match(self, match_text: str, metric_name: str) -> str:
        """Extract unit from matched text - UPDATED WITH NEW METRICS"""
        # Map metric names to default units - UPDATED WITH NEW METRICS
        default_units = {
            'ghg_emissions': 'tCO2e',
            'scope1_emissions': 'tCO2e',
            'scope2_emissions': 'tCO2e',
            'scope3_emissions': 'tCO2e',
            'co2_emissions': 'tCO2e',
            'energy_consumption': 'MWh',
            'renewable_energy': '%',
            'water_withdrawal': 'm³',
            'waste_recycled': '%',
            'hazardous_waste': 'tons',
            'employee_turnover': '%',
            'female_representation': '%',
            'training_hours': 'hours',
            'lost_time_injury': 'rate',
            'employee_satisfaction': '%',
            'community_investment': 'USD',
            'board_independence': '%',
            'female_directors': '%',
            'ceo_pay_ratio': 'ratio',
            'ethics_training': '%',
            'whistleblower_cases': 'count'
        }
        
        # Check for units in the match text - ENHANCED
        unit_patterns = {
            'tCO2e': ['tco2e', 'tons co2', 'tonnes co2', 't co2', 'mtco2e', 'ktco2e'],
            '%': ['%', 'percent', 'percentage', 'pct'],
            'MWh': ['mwh', 'megawatt-hour', 'megawatt hour', 'gwh', 'kwh'],
            'm³': ['m³', 'cubic meter', 'cubic meters', 'm3', 'kl', 'kiloliter'],
            'hours': ['hour', 'hrs', 'hr', 'hours'],
            'rate': ['rate', 'ratio', 'frequency'],
            'USD': ['usd', '$', 'dollar', 'dollars', 'million', 'billion'],
            'tons': ['tons', 'tonnes', 't', 'metric tons'],
            'ratio': ['ratio', ':1', 'times', 'x']
        }
        
        match_lower = match_text.lower()
        for unit, patterns in unit_patterns.items():
            if any(pattern in match_lower for pattern in patterns):
                return unit
        
        # Return default if no unit found
        return default_units.get(metric_name, 'unknown')
    
    def _select_best_value(self, values: List[Dict], metric_name: str) -> Dict:
        """Select the best value from multiple matches"""
        if not values:
            return None
        
        if len(values) == 1:
            return values[0]
        
        # Sort by confidence
        values.sort(key=lambda x: x['confidence'], reverse=True)
        
        # For certain metrics, prefer specific patterns
        if metric_name in ['ghg_emissions', 'scope1_emissions', 'scope2_emissions', 
                          'scope3_emissions', 'co2_emissions', 'energy_consumption']:
            # Prefer values with "total" or from executive summary
            for v in values:
                if 'total' in v['context'] or 'executive' in v['context']:
                    return v
        
        # Return the highest confidence value
        return values[0]
    
    def get_metric_category(self, metric_name: str) -> str:
        """Determine which ESG category a metric belongs to"""
        return self._category_map.get(metric_name, 'unknown')
    
    def calculate_metric_score(self, metric_name: str, value: float) -> Tuple[float, str]:
        """
        Calculate score for a specific metric (0-100)
        Returns score and explanation
        """
        category = self.get_metric_category(metric_name)
        
        if category == 'unknown':
            return 50, "Metric not in scoring framework"
        
        # Get norms for this category
        if category == 'environmental':
            norms = self.current_norms.ENVIRONMENTAL
        elif category == 'social':
            norms = self.current_norms.SOCIAL
        elif category == 'governance':
            norms = self.current_norms.GOVERNANCE
        else:
            return 50, "Category not found"
        
        if metric_name not in norms:
            return 50, "Metric not in scoring framework"
        
        metric_info = norms[metric_name]
        ideal = metric_info['ideal']
        
        # Convert value to float for comparison
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return 50, f"Cannot score non-numeric value: {value}"
        
        # Handle different ideal types
        if ideal == 'decreasing':
            # Lower is better - use threshold for scaling
            threshold = metric_info.get('threshold', 1000)
            if numeric_value <= 0:
                score = 100
            else:
                # Score decreases as value increases relative to threshold
                ratio = min(numeric_value / threshold, 2.0)  # Cap at 2x threshold
                score = max(0, 100 - (ratio * 50))
            explanation = f"Current: {self._format_number(numeric_value)} {metric_info['unit']}. Lower values are better."
        
        elif ideal == 'increasing':
            # Higher is better
            threshold = metric_info.get('threshold', 1000000)
            if numeric_value >= threshold:
                score = 100
            else:
                score = min(100, (numeric_value / threshold) * 100)
            explanation = f"Current: {self._format_number(numeric_value)} {metric_info['unit']}. Higher values are better."
        
        elif isinstance(ideal, tuple):
            # Range is ideal
            min_val, max_val = ideal
            
            if min_val <= numeric_value <= max_val:
                score = 100
                explanation = f"Current: {self._format_number(numeric_value)} {metric_info['unit']}. Within ideal range ({min_val}-{max_val})."
            elif numeric_value < min_val:
                # Score decreases as value moves further below ideal
                if min_val > 0:
                    score = max(0, (numeric_value / min_val) * 100)
                else:
                    score = 0
                explanation = f"Current: {self._format_number(numeric_value)} {metric_info['unit']}. Below ideal minimum ({min_val})."
            else:  # numeric_value > max_val
                if max_val > 0:
                    overshoot = (numeric_value - max_val) / max_val
                    score = max(0, 100 - (overshoot * 100))
                else:
                    score = 0
                explanation = f"Current: {self._format_number(numeric_value)} {metric_info['unit']}. Above ideal maximum ({max_val})."
        
        else:
            score = 50
            explanation = "Scoring logic not defined for this metric"
        
        # Apply confidence adjustment
        return min(100, max(0, round(score, 1))), explanation
    
    def _format_number(self, num: float) -> str:
        """Format number with commas for readability"""
        try:
            if isinstance(num, (int, float)):
                if abs(num) >= 1000000:
                    return f"{num/1000000:.1f}M"
                elif abs(num) >= 1000:
                    return f"{num/1000:.1f}K"
                else:
                    return f"{num:,.0f}" if num.is_integer() else f"{num:,.2f}"
            else:
                return str(num)
        except:
            return str(num)
    
    def calculate_category_scores(self, extracted_metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Calculate scores for each ESG category
        """
        scores = {
            'environmental': {'score': 0, 'metrics': [], 'weighted_score': 0},
            'social': {'score': 0, 'metrics': [], 'weighted_score': 0},
            'governance': {'score': 0, 'metrics': [], 'weighted_score': 0}
        }
        
        category_weights = {'environmental': 0.4, 'social': 0.35, 'governance': 0.25}
        
        for category in scores.keys():
            category_metrics = []
            total_weight = 0
            weighted_score_sum = 0
            
            # Get norms for this category
            if category == 'environmental':
                norms = self.current_norms.ENVIRONMENTAL
            elif category == 'social':
                norms = self.current_norms.SOCIAL
            else:  # governance
                norms = self.current_norms.GOVERNANCE
            
            for metric_name, metric_info in extracted_metrics.items():
                if self.get_metric_category(metric_name) == category:
                    value = metric_info['value']
                    if isinstance(value, (int, float)):
                        score, explanation = self.calculate_metric_score(metric_name, value)
                        
                        # Get weight from norms
                        weight = norms.get(metric_name, {}).get('weight', 1.0/len(norms) if norms else 0)
                        
                        category_metrics.append({
                            'metric': metric_name,
                            'value': value,
                            'unit': metric_info.get('unit', 'unknown'),
                            'score': score,
                            'weight': weight,
                            'explanation': explanation,
                            'confidence': metric_info.get('confidence', 0.5)
                        })
                        
                        total_weight += weight
                        weighted_score_sum += score * weight
            
            if category_metrics and total_weight > 0:
                category_score = weighted_score_sum / total_weight
            else:
                # If no metrics found, use default score
                category_score = 50
            
            scores[category]['score'] = round(category_score, 1)
            scores[category]['metrics'] = category_metrics
            scores[category]['weighted_score'] = round(category_score * category_weights[category], 1)
        
        # Calculate overall score
        overall_score = sum(scores[cat]['weighted_score'] for cat in ['environmental', 'social', 'governance'])
        scores['overall'] = {'score': round(overall_score, 1)}
        
        return scores
    
    def generate_recommendations(self, extracted_metrics: Dict[str, Dict], scores: Dict) -> List[Dict]:
        """
        Generate specific recommendations to improve ESG scores - UPDATED WITH NEW METRICS
        """
        recommendations = []
        
        # Organize metrics by category
        env_metrics = {}
        social_metrics = {}
        gov_metrics = {}
        
        for metric_name, metric_info in extracted_metrics.items():
            category = self.get_metric_category(metric_name)
            if category == 'environmental':
                env_metrics[metric_name] = metric_info
            elif category == 'social':
                social_metrics[metric_name] = metric_info
            elif category == 'governance':
                gov_metrics[metric_name] = metric_info
        
        # Get scored metrics for each category
        scored_env = {m['metric']: m for m in scores['environmental']['metrics']}
        scored_social = {m['metric']: m for m in scores['social']['metrics']}
        scored_gov = {m['metric']: m for m in scores['governance']['metrics']}
        
        # Environmental recommendations - UPDATED WITH SCOPE RECOMMENDATIONS
        if 'ghg_emissions' in scored_env:
            metric = scored_env['ghg_emissions']
            if metric['score'] < 60:
                recommendations.append({
                    'category': 'Environmental',
                    'priority': 'High',
                    'metric': 'GHG Emissions',
                    'current': f"{self._format_number(metric['value'])} {metric['unit']}",
                    'score': metric['score'],
                    'recommendation': 'Implement comprehensive carbon reduction strategy',
                    'action_items': [
                        'Conduct detailed carbon footprint assessment across Scope 1, 2, and 3',
                        'Set science-based targets aligned with Paris Agreement',
                        'Invest in energy efficiency upgrades',
                        'Transition to renewable energy sources',
                        'Implement carbon offset program for unavoidable emissions'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        # NEW: Scope 1 recommendation
        if 'scope1_emissions' in scored_env:
            metric = scored_env['scope1_emissions']
            if metric['score'] < 60:
                recommendations.append({
                    'category': 'Environmental',
                    'priority': 'High',
                    'metric': 'Scope 1 Emissions',
                    'current': f"{self._format_number(metric['value'])} {metric['unit']}",
                    'score': metric['score'],
                    'recommendation': 'Reduce direct emissions from owned sources',
                    'action_items': [
                        'Upgrade equipment to energy-efficient models',
                        'Switch to low-carbon fuels',
                        'Implement fugitive emission controls',
                        'Optimize production processes',
                        'Electrify fleet vehicles'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        # NEW: Scope 2 recommendation
        if 'scope2_emissions' in scored_env:
            metric = scored_env['scope2_emissions']
            if metric['score'] < 60:
                recommendations.append({
                    'category': 'Environmental',
                    'priority': 'Medium',
                    'metric': 'Scope 2 Emissions',
                    'current': f"{self._format_number(metric['value'])} {metric['unit']}",
                    'score': metric['score'],
                    'recommendation': 'Reduce indirect emissions from purchased energy',
                    'action_items': [
                        'Purchase Renewable Energy Certificates (RECs)',
                        'Enter into Power Purchase Agreements (PPAs)',
                        'Install on-site renewable generation',
                        'Improve energy efficiency in facilities',
                        'Choose green energy tariffs from utilities'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        # NEW: Scope 3 recommendation
        if 'scope3_emissions' in scored_env:
            metric = scored_env['scope3_emissions']
            if metric['score'] < 50:
                recommendations.append({
                    'category': 'Environmental',
                    'priority': 'Medium',
                    'metric': 'Scope 3 Emissions',
                    'current': f"{self._format_number(metric['value'])} {metric['unit']}",
                    'score': metric['score'],
                    'recommendation': 'Address value chain emissions',
                    'action_items': [
                        'Engage suppliers on carbon reduction',
                        'Optimize logistics and transportation',
                        'Reduce business travel',
                        'Design products for lower carbon footprint',
                        'Work with customers on emissions reduction'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        if 'renewable_energy' in scored_env:
            metric = scored_env['renewable_energy']
            if metric['score'] < 50:
                recommendations.append({
                    'category': 'Environmental',
                    'priority': 'Medium',
                    'metric': 'Renewable Energy Usage',
                    'current': f"{metric['value']}%",
                    'score': metric['score'],
                    'recommendation': 'Increase proportion of renewable energy in energy mix',
                    'action_items': [
                        'Install on-site solar panels or wind turbines',
                        'Purchase Renewable Energy Certificates (RECs)',
                        'Enter into Power Purchase Agreements (PPAs) with renewable providers',
                        'Join industry initiatives like RE100',
                        'Implement energy storage solutions'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        # Social recommendations
        if 'female_representation' in scored_social:
            metric = scored_social['female_representation']
            if metric['score'] < 60:
                recommendations.append({
                    'category': 'Social',
                    'priority': 'High',
                    'metric': 'Gender Diversity',
                    'current': f"{metric['value']}%",
                    'score': metric['score'],
                    'recommendation': 'Enhance gender diversity and inclusion initiatives',
                    'action_items': [
                        'Implement blind recruitment processes',
                        'Set and publicly disclose diversity targets',
                        'Create leadership development programs for women',
                        'Conduct regular pay equity audits',
                        'Establish employee resource groups for women'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        if 'employee_turnover' in scored_social:
            metric = scored_social['employee_turnover']
            if metric['score'] < 60:
                recommendations.append({
                    'category': 'Social',
                    'priority': 'Medium',
                    'metric': 'Employee Retention',
                    'current': f"{metric['value']}%",
                    'score': metric['score'],
                    'recommendation': 'Improve employee retention and satisfaction',
                    'action_items': [
                        'Conduct stay interviews to understand retention drivers',
                        'Enhance career development and progression opportunities',
                        'Improve work-life balance through flexible arrangements',
                        'Regularly benchmark compensation against market',
                        'Strengthen leadership development programs'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        # Governance recommendations
        if 'board_independence' in scored_gov:
            metric = scored_gov['board_independence']
            if metric['score'] < 70:
                recommendations.append({
                    'category': 'Governance',
                    'priority': 'Medium',
                    'metric': 'Board Independence',
                    'current': f"{metric['value']}%",
                    'score': metric['score'],
                    'recommendation': 'Strengthen board independence and oversight',
                    'action_items': [
                        'Increase proportion of independent directors',
                        'Ensure audit committee is fully independent',
                        'Separate CEO and Chair roles',
                        'Enhance board diversity across multiple dimensions',
                        'Implement regular board effectiveness reviews'
                    ],
                    'potential_improvement': f"Potential score increase: {100 - metric['score']:.0f} points"
                })
        
        # Overall recommendations based on total score
        overall_score = scores['overall']['score']
        if overall_score < 60:
            recommendations.append({
                'category': 'Overall',
                'priority': 'High',
                'metric': 'ESG Program Maturity',
                'current': f"Overall Score: {overall_score}/100",
                'score': overall_score,
                'recommendation': 'Develop comprehensive ESG governance and strategy',
                'action_items': [
                    'Establish board-level ESG committee with clear mandate',
                    'Develop and publish comprehensive ESG policy framework',
                    'Set measurable, time-bound ESG targets aligned with material issues',
                    'Implement robust ESG data collection and verification systems',
                    'Enhance ESG disclosure and reporting practices',
                    'Integrate ESG metrics into executive compensation'
                ],
                'potential_improvement': f"Potential score increase: {100 - overall_score:.0f} points"
            })
        
        # Sort by priority (High first) and score (lowest first)
        recommendations.sort(key=lambda x: (0 if x['priority'] == 'High' else 1, x['score']))
        
        return recommendations
    
    def analyze_report(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete analysis of an ESG report
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING ESG REPORT: {os.path.basename(pdf_path)}")
        print(f"Industry: {self.industry.title()}")
        print(f"{'='*60}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("\n❌ Failed to extract text from PDF.")
            print("Possible solutions:")
            print("  1. Ensure the PDF contains text (not just scanned images)")
            print("  2. Try OCR preprocessing if it's a scanned document")
            print("  3. Check if the PDF is corrupted or password-protected")
            return None
            
        print(f"✓ Extracted {len(text):,} characters from PDF")
        
        # Extract specific metrics
        extracted_metrics = self.extract_specific_metrics(text)
        print(f"✓ Found {len(extracted_metrics)} specific ESG metrics")
        
        if len(extracted_metrics) == 0:
            print("\n⚠ Warning: No ESG metrics were automatically extracted.")
            print("This could mean:")
            print("  1. The PDF doesn't contain standard ESG metrics")
            print("  2. The metrics are in a non-standard format")
            print("  3. The text extraction didn't capture the relevant sections")
        
        # Calculate scores
        scores = self.calculate_category_scores(extracted_metrics)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(extracted_metrics, scores)
        
        # Prepare results
        results = {
            'company': os.path.basename(pdf_path).replace('.pdf', ''),
            'industry': self.industry,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'extracted_metrics': extracted_metrics,
            'esg_scores': scores,
            'recommendations': recommendations,
            'analysis_summary': self._create_summary(scores, extracted_metrics, recommendations)
        }
        
        return results
    
    def _create_summary(self, scores: Dict, metrics: Dict, recommendations: List) -> Dict:
        """Create analysis summary"""
        high_priority = len([r for r in recommendations if r['priority'] == 'High'])
        medium_priority = len([r for r in recommendations if r['priority'] == 'Medium'])
        
        # Calculate average improvement potential
        if recommendations:
            avg_improvement = np.mean([100 - r['score'] for r in recommendations])
        else:
            avg_improvement = 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_metrics_extracted': len(metrics),
            'overall_esg_score': scores['overall']['score'],
            'category_scores': {
                'environmental': scores['environmental']['score'],
                'social': scores['social']['score'],
                'governance': scores['governance']['score']
            },
            'recommendation_count': len(recommendations),
            'high_priority_recommendations': high_priority,
            'medium_priority_recommendations': medium_priority,
            'average_improvement_potential': round(avg_improvement, 1)
        }
    
    def validate_accuracy(self, test_cases: List[TestCase] = None) -> Dict:
        """
        Validate the accuracy of metric extraction
        """
        validator = ESGAccuracyValidator(self)
        results = validator.run_validation(test_cases)
        validator.generate_validation_report(results, "esg_accuracy_report")
        return results
    
    def benchmark_performance(self, sample_texts: List[str]) -> Dict:
        """
        Benchmark extraction performance on sample texts
        """
        benchmark_results = {
            'texts_processed': 0,
            'metrics_extracted': 0,
            'extraction_time': 0,
            'by_metric': {}
        }
        
        import time
        start_time = time.time()
        
        for text in sample_texts:
            extracted = self.extract_specific_metrics(text)
            benchmark_results['metrics_extracted'] += len(extracted)
            
            for metric_name, metric_info in extracted.items():
                if metric_name not in benchmark_results['by_metric']:
                    benchmark_results['by_metric'][metric_name] = {
                        'count': 0,
                        'confidence_sum': 0
                    }
                benchmark_results['by_metric'][metric_name]['count'] += 1
                benchmark_results['by_metric'][metric_name]['confidence_sum'] += metric_info.get('confidence', 0.5)
        
        benchmark_results['extraction_time'] = time.time() - start_time
        benchmark_results['texts_processed'] = len(sample_texts)
        
        # Calculate averages
        if benchmark_results['texts_processed'] > 0:
            benchmark_results['avg_metrics_per_text'] = (
                benchmark_results['metrics_extracted'] / benchmark_results['texts_processed']
            )
            benchmark_results['avg_extraction_time'] = (
                benchmark_results['extraction_time'] / benchmark_results['texts_processed']
            )
        
        # Calculate average confidence per metric
        for metric_name, stats in benchmark_results['by_metric'].items():
            if stats['count'] > 0:
                stats['avg_confidence'] = stats['confidence_sum'] / stats['count']
        
        return benchmark_results

class ESGReportExporter:
    """Handles exporting analysis results to various formats"""
    
    @staticmethod
    def export_to_csv(analysis_results: Dict, filename: str = "esg_analysis.csv"):
        """Export analysis results to CSV files"""
        
        if not analysis_results:
            print("❌ No analysis results to export")
            return None
        
        # Create metrics DataFrame
        metrics_data = []
        analyzer = analysis_results.get('analyzer', ESGReportAnalyzer())
        
        for metric_name, metric_info in analysis_results['extracted_metrics'].items():
            category = analyzer.get_metric_category(metric_name)
            metrics_data.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Value': metric_info['value'],
                'Unit': metric_info.get('unit', 'unknown'),
                'Category': category.title() if category != 'unknown' else 'Unknown',
                'Confidence': metric_info.get('confidence', 0.5)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create scores DataFrame
        scores_data = []
        for category in ['environmental', 'social', 'governance', 'overall']:
            if category in analysis_results['esg_scores']:
                scores_data.append({
                    'Category': category.title(),
                    'Score': analysis_results['esg_scores'][category]['score']
                })
        
        scores_df = pd.DataFrame(scores_data)
        
        # Create recommendations DataFrame
        rec_data = []
        for rec in analysis_results['recommendations']:
            rec_data.append({
                'Category': rec['category'],
                'Priority': rec['priority'],
                'Metric': rec['metric'],
                'Current_Value': rec['current'],
                'Score': rec['score'],
                'Recommendation': rec['recommendation'],
                'Action_Items': ' | '.join(rec['action_items']),
                'Potential_Improvement': rec.get('potential_improvement', 'N/A')
            })
        
        rec_df = pd.DataFrame(rec_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Export to separate CSV files
        base_name = filename.replace('.csv', '')
        
        metrics_df.to_csv(f"{base_name}_metrics.csv", index=False)
        scores_df.to_csv(f"{base_name}_scores.csv", index=False)
        rec_df.to_csv(f"{base_name}_recommendations.csv", index=False)
        
        print(f"✓ Metrics exported to: {base_name}_metrics.csv")
        print(f"✓ Scores exported to: {base_name}_scores.csv")
        print(f"✓ Recommendations exported to: {base_name}_recommendations.csv")
        
        return {
            'metrics_file': f"{base_name}_metrics.csv",
            'scores_file': f"{base_name}_scores.csv",
            'recommendations_file': f"{base_name}_recommendations.csv"
        }
    
    @staticmethod
    def export_to_json(analysis_results: Dict, filename: str = "esg_analysis.json"):
        """Export analysis results to JSON"""
        
        if not analysis_results:
            print("❌ No analysis results to export")
            return None
            
        # Remove analyzer object if present (not JSON serializable)
        export_data = analysis_results.copy()
        export_data.pop('analyzer', None)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✓ Full analysis exported to: {filename}")
        return filename
    
    @staticmethod
    def export_summary_report(analysis_results: Dict, filename: str = "esg_summary.txt"):
        """Export a text summary report"""
        
        if not analysis_results:
            print("❌ No analysis results to export")
            return None
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ESG ANALYSIS SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Company: {analysis_results.get('company', 'Unknown')}\n")
            f.write(f"Industry: {analysis_results.get('industry', 'Unknown').title()}\n")
            f.write(f"Analysis Date: {analysis_results.get('analysis_date', 'Unknown')}\n\n")
            
            f.write("OVERALL ESG SCORE\n")
            f.write("-"*40 + "\n")
            overall_score = analysis_results['esg_scores']['overall']['score']
            f.write(f"Score: {overall_score}/100\n\n")
            
            f.write("CATEGORY SCORES\n")
            f.write("-"*40 + "\n")
            for category in ['environmental', 'social', 'governance']:
                score = analysis_results['esg_scores'][category]['score']
                f.write(f"{category.title()}: {score}/100\n")
            f.write("\n")
            
            f.write("KEY METRICS EXTRACTED\n")
            f.write("-"*40 + "\n")
            for metric_name, metric_info in list(analysis_results['extracted_metrics'].items())[:10]:  # Top 10
                value = metric_info['value']
                unit = metric_info.get('unit', '')
                f.write(f"• {metric_name.replace('_', ' ').title()}: {value} {unit}\n")
            f.write("\n")
            
            f.write("TOP RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            for i, rec in enumerate(analysis_results['recommendations'][:5], 1):  # Top 5
                f.write(f"{i}. [{rec['priority']}] {rec['recommendation']}\n")
                f.write(f"   Current: {rec['current']} (Score: {rec['score']}/100)\n\n")
        
        print(f"✓ Summary report exported to: {filename}")
        return filename

def print_detailed_report(analysis_results: Dict):
    """
    Print detailed analysis report
    """
    if not analysis_results:
        print("❌ No analysis results to display")
        return
        
    print("\n" + "="*60)
    print("ESG ANALYSIS REPORT")
    print("="*60)
    
    # Company and industry info
    print(f"\n🏢 COMPANY: {analysis_results.get('company', 'Unknown')}")
    print(f"🏭 INDUSTRY: {analysis_results.get('industry', 'Unknown').title()}")
    print(f"📅 ANALYSIS DATE: {analysis_results.get('analysis_date', 'Unknown')}")
    
    # Overall Score with rating
    overall_score = analysis_results['esg_scores']['overall']['score']
    if overall_score >= 80:
        rating = "LEADER 🏆"
    elif overall_score >= 60:
        rating = "ADVANCED ⭐"
    elif overall_score >= 40:
        rating = "DEVELOPING 📈"
    else:
        rating = "BEGINNER 🔄"
    
    print(f"\n📊 OVERALL ESG SCORE: {overall_score}/100 - {rating}")
    
    # Category Scores
    print("\n📈 CATEGORY SCORES:")
    for category in ['environmental', 'social', 'governance']:
        score = analysis_results['esg_scores'][category]['score']
        metrics = analysis_results['esg_scores'][category]['metrics']
        
        print(f"\n  • {category.upper()}: {score}/100")
        
        if metrics:
            print(f"    Key Metrics Found:")
            for metric in metrics[:3]:  # Show top 3 metrics
                metric_name = metric['metric'].replace('_', ' ').title()
                print(f"      - {metric_name}: {metric['value']} {metric['unit']} (Score: {metric['score']}/100)")
    
    # Extracted Metrics
    print(f"\n📋 DETAILED METRICS EXTRACTED ({len(analysis_results['extracted_metrics'])} total):")
    for i, (metric_name, metric_info) in enumerate(analysis_results['extracted_metrics'].items(), 1):
        if i <= 15:  # Limit display to 15 metrics
            formatted_name = metric_name.replace('_', ' ').title()
            print(f"  {i:2d}. {formatted_name}: {metric_info['value']} {metric_info.get('unit', '')}")
    
    if len(analysis_results['extracted_metrics']) > 15:
        print(f"  ... and {len(analysis_results['extracted_metrics']) - 15} more metrics")
    
    # Recommendations
    print(f"\n🎯 ACTIONABLE RECOMMENDATIONS ({len(analysis_results['recommendations'])} total):")
    
    high_priority = [r for r in analysis_results['recommendations'] if r['priority'] == 'High']
    medium_priority = [r for r in analysis_results['recommendations'] if r['priority'] == 'Medium']
    
    if high_priority:
        print(f"\n  🔴 HIGH PRIORITY ({len(high_priority)}):")
        for rec in high_priority:
            print(f"\n    • {rec['metric']}")
            print(f"      Current: {rec['current']} | Score: {rec['score']}/100")
            print(f"      {rec['recommendation']}")
            print(f"      Key Actions:")
            for action in rec['action_items'][:3]:  # Show top 3 actions
                print(f"        - {action}")
    
    if medium_priority:
        print(f"\n  🟡 MEDIUM PRIORITY ({len(medium_priority)}):")
        for rec in medium_priority[:3]:  # Show top 3 medium priority
            print(f"\n    • {rec['metric']}")
            print(f"      Current: {rec['current']} | Score: {rec['score']}/100")
            print(f"      {rec['recommendation']}")
    
    # Summary Statistics
    summary = analysis_results['analysis_summary']
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"  • Total Metrics Extracted: {summary['total_metrics_extracted']}")
    print(f"  • High Priority Recommendations: {summary['high_priority_recommendations']}")
    print(f"  • Medium Priority Recommendations: {summary['medium_priority_recommendations']}")
    print(f"  • Average Improvement Potential: {summary['average_improvement_potential']:.1f} points")
    
    # Estimated time to improve
    if high_priority:
        est_months = len(high_priority) * 3 + len(medium_priority) * 6
        print(f"  • Estimated Time to Significant Improvement: {est_months} months")
    
    print("\n" + "="*60)

def demonstrate_with_sample_data():
    """
    Demonstrate the analyzer with sample ESG data
    """
    print("\n" + "="*60)
    print("SAMPLE ESG ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create analyzer for manufacturing industry
    analyzer = ESGReportAnalyzer(industry_norm='manufacturing')
    
    # First, validate accuracy
    print("\n🔍 VALIDATING EXTRACTION ACCURACY...")
    accuracy_results = analyzer.validate_accuracy()
    
    # Sample metrics from a manufacturing company report - UPDATED WITH SCOPE AND CO2
    sample_metrics = {
        'ghg_emissions': {'value': 125000, 'unit': 'tCO2e', 'confidence': 0.8},
        'scope1_emissions': {'value': 45000, 'unit': 'tCO2e', 'confidence': 0.8},
        'scope2_emissions': {'value': 35000, 'unit': 'tCO2e', 'confidence': 0.8},
        'scope3_emissions': {'value': 45000, 'unit': 'tCO2e', 'confidence': 0.7},
        'co2_emissions': {'value': 115000, 'unit': 'tCO2e', 'confidence': 0.8},
        'energy_consumption': {'value': 85000, 'unit': 'MWh', 'confidence': 0.7},
        'renewable_energy': {'value': 15, 'unit': '%', 'confidence': 0.9},
        'water_withdrawal': {'value': 2500000, 'unit': 'm³', 'confidence': 0.6},
        'waste_recycled': {'value': 45, 'unit': '%', 'confidence': 0.7},
        'employee_turnover': {'value': 18, 'unit': '%', 'confidence': 0.8},
        'female_representation': {'value': 28, 'unit': '%', 'confidence': 0.9},
        'training_hours': {'value': 12, 'unit': 'hours', 'confidence': 0.6},
        'board_independence': {'value': 45, 'unit': '%', 'confidence': 0.8},
        'female_directors': {'value': 20, 'unit': '%', 'confidence': 0.9},
        'ceo_pay_ratio': {'value': 120, 'unit': 'ratio', 'confidence': 0.7}
    }
    
    # Calculate scores
    scores = analyzer.calculate_category_scores(sample_metrics)
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations(sample_metrics, scores)
    
    # Prepare results
    results = {
        'company': 'Sample Manufacturing Corp',
        'industry': 'manufacturing',
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'extracted_metrics': sample_metrics,
        'esg_scores': scores,
        'recommendations': recommendations,
        'analysis_summary': analyzer._create_summary(scores, sample_metrics, recommendations),
        'analyzer': analyzer  # Include analyzer for category lookups
    }
    
    # Print report
    print_detailed_report(results)
    
    # Export sample data
    exporter = ESGReportExporter()
    exporter.export_to_csv(results, "sample_esg_analysis.csv")
    exporter.export_summary_report(results, "sample_esg_summary.txt")
    
    return results

def analyze_real_report(pdf_path: str, industry: str = 'general'):
    """
    Analyze a real ESG report PDF
    """
    print(f"\n🔍 Starting analysis of: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return None
    
    # Initialize analyzer
    analyzer = ESGReportAnalyzer(industry_norm=industry)
    
    # First validate accuracy
    print("\n🔍 VALIDATING EXTRACTION ACCURACY...")
    accuracy_results = analyzer.validate_accuracy()
    
    # Check if accuracy meets threshold
    overall_accuracy = accuracy_results['extraction_accuracy']['overall_accuracy']
    if overall_accuracy < 80:
        print(f"\n⚠ WARNING: Extraction accuracy is {overall_accuracy}% (below 80% target)")
        proceed = input("Continue with analysis anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Analysis cancelled.")
            return None
    
    try:
        # Analyze the report
        results = analyzer.analyze_report(pdf_path)
        
        if not results:
            print("\n❌ Analysis failed - no results generated")
            return None
            
        results['analyzer'] = analyzer  # Add analyzer to results
        
        # Print detailed report
        print_detailed_report(results)
        
        # Export results
        exporter = ESGReportExporter()
        base_name = os.path.basename(pdf_path).replace('.pdf', '')
        
        csv_files = exporter.export_to_csv(results, f"{base_name}_analysis.csv")
        json_file = exporter.export_to_json(results, f"{base_name}_analysis.json")
        summary_file = exporter.export_summary_report(results, f"{base_name}_summary.txt")
        
        print(f"\n✅ Analysis complete! Files saved:")
        if csv_files:
            print(f"   - {csv_files['metrics_file']}")
            print(f"   - {csv_files['scores_file']}")
            print(f"   - {csv_files['recommendations_file']}")
        if json_file:
            print(f"   - {json_file}")
        if summary_file:
            print(f"   - {summary_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing report: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_accuracy_test():
    """Run comprehensive accuracy testing"""
    print("\n" + "="*60)
    print("COMPREHENSIVE ACCURACY TESTING")
    print("="*60)
    
    analyzer = ESGReportAnalyzer(industry_norm='general')
    validator = ESGAccuracyValidator(analyzer)
    
    # Run validation
    results = validator.run_validation()
    
    # Generate detailed report
    validator.generate_validation_report(results, "accuracy_test_results")
    
    # Check if accuracy meets target
    target_accuracy = 80
    overall_accuracy = results['extraction_accuracy']['overall_accuracy']
    
    if overall_accuracy >= target_accuracy:
        print(f"\n🎉 SUCCESS: Accuracy {overall_accuracy}% meets {target_accuracy}% target")
    else:
        print(f"\n⚠ WARNING: Accuracy {overall_accuracy}% below {target_accuracy}% target")
        
        # Show improvement suggestions
        print("\nImprovement suggestions:")
        
        # Check which categories need improvement
        for category, stats in results['category_accuracy'].items():
            if stats['accuracy'] < target_accuracy:
                print(f"  • {category.capitalize()} category: {stats['accuracy']}% accuracy")
        
        # Check extraction rate
        if results['extraction_accuracy']['extraction_rate'] < target_accuracy:
            print("  • Improve regex patterns to capture more metrics")
        
        # Check value accuracy
        if results['extraction_accuracy']['value_accuracy'] < target_accuracy:
            print("  • Improve value parsing and unit conversion")
    
    return results

def main():
    """
    Main function to run ESG analysis
    """
    print("="*60)
    print("           ESG REPORT ANALYZER")
    print("="*60)
    print("\nOptions:")
    print("1. Analyze a real ESG report PDF")
    print("2. Run sample analysis demonstration")
    print("3. Run comprehensive accuracy testing")
    print("4. Quick test with sample text")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        pdf_path = input("\nEnter path to ESG report PDF: ").strip()
        if not pdf_path.endswith('.pdf'):
            pdf_path = pdf_path + '.pdf'
        
        print("\nAvailable industries:")
        industries = ['general', 'manufacturing', 'mining', 'technology', 'finance']
        for i, ind in enumerate(industries, 1):
            print(f"  {i}. {ind.title()}")
        
        ind_choice = input("\nSelect industry (1-5, default=1): ").strip()
        if ind_choice.isdigit() and 1 <= int(ind_choice) <= 5:
            industry = industries[int(ind_choice) - 1]
        else:
            industry = 'general'
        
        analyze_real_report(pdf_path, industry)
    
    elif choice == '2':
        demonstrate_with_sample_data()
    
    elif choice == '3':
        run_accuracy_test()
    
    elif choice == '4':
        # Quick test with sample text - UPDATED WITH SCOPE AND CO2
        analyzer = ESGReportAnalyzer()
        test_text = """
        Our company reduced GHG emissions to 50,000 tCO2e last year.
        Scope 1 emissions were 20,000 tCO2e, Scope 2 were 15,000 tCO2e, and Scope 3 were 15,000 tCO2e.
        CO2 emissions decreased to 45,000 tonnes.
        Renewable energy usage increased to 45%.
        Female representation in leadership is 35%.
        """
        print("\nTesting extraction with sample text...")
        extracted = analyzer.extract_specific_metrics(test_text)
        print(f"Extracted metrics: {len(extracted)}")
        for metric, info in extracted.items():
            print(f"  - {metric}: {info['value']} {info.get('unit', '')}")
    
    elif choice == '5':
        print("\nGoodbye!")
    
    else:
        print("\nInvalid choice. Please run again.")

if __name__ == "__main__":
    main()
