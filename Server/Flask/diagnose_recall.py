"""
Lightweight diagnostic — analyze existing labeled data for coverage gaps.
"""
import os, sys, re, json
from collections import Counter

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

labeled_path = os.path.join(os.path.dirname(__file__), 'ml_pipeline', 'data', 'esg_labeled.jsonl')

if not os.path.exists(labeled_path):
    print(f"Not found: {labeled_path}")
    sys.exit(1)

samples = []
with open(labeled_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line.strip()))

positives = [s for s in samples if s['metric_name'] != 'no_metric']
negatives = [s for s in samples if s['metric_name'] == 'no_metric']

print(f"Total: {len(samples)}, Positives: {len(positives)}, Negatives: {len(negatives)}")

dist = Counter(s['metric_name'] for s in positives)
print(f"\nMetric distribution:")
for k, v in sorted(dist.items(), key=lambda x: x[1]):
    print(f"  {k}: {v}")

# ── ANALYZE NEGATIVE SAMPLES FOR NEAR-MISSES ──
esg_keywords = [
    'emission', 'scope', 'energy', 'water', 'waste', 'turnover',
    'training', 'board', 'director', 'ethics', 'compliance',
    'renewable', 'carbon', 'co2', 'ghg', 'hazardous', 'injury',
    'safety', 'satisfaction', 'csr', 'community', 'whistleblower',
    'attrition', 'female', 'women', 'gender', 'diversity',
    'grievance', 'recycl', 'independence', 'tco2', 'mwh', 'gwh',
    'employee', 'workforce', 'percentage', 'cubic', 'tonnes',
    'metric ton', 'consumption', 'discharge', 'intensity',
    'biodiversity', 'pollution', 'effluent', 'spill', 'fatality',
]

near_misses = []
for s in negatives:
    text_lower = s['text'].lower()
    has_number = bool(re.search(r'\d{2,}', text_lower))
    matching_kws = [kw for kw in esg_keywords if kw in text_lower]

    if has_number and len(matching_kws) >= 2:
        near_misses.append({
            'text': s['text'][:600],
            'keywords': matching_kws[:8],
            'source': s.get('source_pdf', '?')
        })

print(f"\nNegatives with 2+ ESG keywords + numbers: {len(near_misses)} / {len(negatives)}")

print(f"\n=== SAMPLE NEAR-MISS NEGATIVES (first 20) ===")
for i, nm in enumerate(near_misses[:20]):
    print(f"\n--- {i+1}. keywords: {nm['keywords']} | src: {nm['source']} ---")
    print(nm['text'][:500])

# ── ANALYZE POSITIVE SAMPLES ── 
print(f"\n=== SAMPLE POSITIVE SAMPLES BY LOW-COUNT METRICS ===")
low_count_metrics = [k for k, v in dist.items() if v <= 20]
for metric in low_count_metrics:
    metric_samples = [s for s in positives if s['metric_name'] == metric][:3]
    print(f"\n--- {metric} (count={dist[metric]}) ---")
    for s in metric_samples:
        val = s.get('value', '?')
        unit = s.get('unit', '?')
        print(f"  value={val} unit={unit}")
        print(f"  text: {s['text'][:250]}")
