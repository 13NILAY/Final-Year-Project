"""Quick test to verify ESG metric extraction fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline.extractor import ml_esg_metric_extractor

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
print("VERIFICATION TEST: ESG Metric Extraction Bug Fixes")
print("=" * 60)

results = ml_esg_metric_extractor(test_text)

print(f"\nExtracted {len(results)} metrics:\n")
for metric, info in sorted(results.items()):
    print(f"  {metric:25s}: {info['value']} {info['unit']}  (conf={info['confidence']:.2f}, src={info['source']})")

# Key bug fix checks
print("\n" + "-" * 60)
print("BUG FIX VERIFICATION:")
print("-" * 60)

# 1. CEO pay ratio should be 85, not 85,000,000
if 'ceo_pay_ratio' in results:
    ceo_val = results['ceo_pay_ratio']['value']
    if ceo_val == 85:
        print(f"  [PASS] CEO pay ratio = {ceo_val} (correct, no million multiplier)")
    else:
        print(f"  [FAIL] CEO pay ratio = {ceo_val} (expected 85)")
else:
    print("  [FAIL] CEO pay ratio not extracted")

# 2. Scope 1 should be 45,000
if 'scope1_emissions' in results:
    s1_val = results['scope1_emissions']['value']
    if s1_val == 45000:
        print(f"  [PASS] Scope 1 emissions = {s1_val} tCO2e (correct)")
    else:
        print(f"  [FAIL] Scope 1 emissions = {s1_val} (expected 45000)")
else:
    print("  [FAIL] Scope 1 emissions not extracted")

# 3. Scope 2 should be 35,000
if 'scope2_emissions' in results:
    s2_val = results['scope2_emissions']['value']
    if s2_val == 35000:
        print(f"  [PASS] Scope 2 emissions = {s2_val} tCO2e (correct)")
    else:
        print(f"  [FAIL] Scope 2 emissions = {s2_val} (expected 35000)")
else:
    print("  [FAIL] Scope 2 emissions not extracted")

# 4. Renewable energy should be 42 (percentage, not scaled)
if 'renewable_energy' in results:
    re_val = results['renewable_energy']['value']
    if re_val == 42:
        print(f"  [PASS] Renewable energy = {re_val}% (correct, no scaling)")
    else:
        print(f"  [FAIL] Renewable energy = {re_val} (expected 42)")
else:
    print("  [FAIL] Renewable energy not extracted")

# 5. Ethics training should be 92 (percentage, not scaled)
if 'ethics_training' in results:
    et_val = results['ethics_training']['value']
    if et_val == 92:
        print(f"  [PASS] Ethics training = {et_val}% (correct, no scaling)")
    else:
        print(f"  [FAIL] Ethics training = {et_val} (expected 92)")
else:
    print("  [FAIL] Ethics training not extracted")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
