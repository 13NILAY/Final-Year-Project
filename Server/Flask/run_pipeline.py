"""
ESG ML Pipeline - Main Entry Point
====================================
Run the full pipeline: Generate Labels → Train Model → Evaluate → Launch API

Usage:
    python run_pipeline.py label       # Step 1: Generate labeled dataset from PDFs
    python run_pipeline.py train       # Step 2: Train the RoBERTa model
    python run_pipeline.py evaluate    # Step 3: Evaluate on test samples (rich)
    python run_pipeline.py analyze <pdf_path> [industry]  # Analyze a PDF
    python run_pipeline.py serve [port] # Launch Flask API
    python run_pipeline.py all         # Run full pipeline (label → train → evaluate)
"""

import os
import sys
import json

# Add the Flask directory to path
FLASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FLASK_DIR)

PIPELINE_DIR = os.path.join(FLASK_DIR, "ml_pipeline")
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
MODELS_DIR = os.path.join(PIPELINE_DIR, "models")
DATASET_DIR = os.path.join(os.path.dirname(FLASK_DIR), "..", "Dataset")
DATASET_DIR = os.path.normpath(DATASET_DIR)

LABELED_DATA_PATH = os.path.join(DATA_DIR, "esg_labeled.jsonl")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pt")


def step_label():
    """Step 1: Generate labeled dataset from ESG PDFs."""
    print("=" * 60)
    print("STEP 1: GENERATING LABELED DATASET")
    print("=" * 60)

    from ml_pipeline.labeling import generate_labeled_dataset

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        print("Expected structure: Dataset/Tech/*.pdf, Dataset/Finance/*.pdf, etc.")
        return False

    print(f"\nDataset directory: {DATASET_DIR}")
    subfolders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"Subfolders found: {subfolders}")
    for sub in subfolders:
        pdf_count = len([f for f in os.listdir(os.path.join(DATASET_DIR, sub)) if f.lower().endswith('.pdf')])
        print(f"  {sub}: {pdf_count} PDFs")

    stats = generate_labeled_dataset(
        pdf_dir=DATASET_DIR,
        output_path=LABELED_DATA_PATH,
        chunk_size=120,
        overlap=40,
        include_negatives=True,
        negative_ratio=0.6,
    )

    print(f"\n✅ Dataset generated: {LABELED_DATA_PATH}")

    # Handle both old and new stats keys
    if 'positive_samples' in stats:
        print(f"   Total labeled samples: {stats['positive_samples']}")
        print(f"   Total negative samples: {stats['negative_samples']}")
    else:
        # Fallback to old keys (for backward compatibility)
        print(f"   Total labeled samples: {stats.get('labeled_chunks', 'N/A')}")
        print(f"   Total negative samples: {stats.get('negative_chunks', 'N/A')}")

    return True
def step_train():
    """Step 2: Train the RoBERTa classifier."""
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING RoBERTa MODEL (Multi-Head Architecture)")
    print("=" * 60)

    from ml_pipeline.labeling import load_dataset
    from ml_pipeline.train import train_model

    if not os.path.exists(LABELED_DATA_PATH):
        print(f"ERROR: Labeled dataset not found: {LABELED_DATA_PATH}")
        print("Run 'python run_pipeline.py label' first.")
        return False

    os.makedirs(MODELS_DIR, exist_ok=True)

    train_data, val_data, test_data = load_dataset(LABELED_DATA_PATH)

    if len(train_data) < 10:
        print(f"WARNING: Only {len(train_data)} training samples.")
        print("Consider adding more PDFs to the Dataset folder.")

    results = train_model(
        train_data=train_data,
        val_data=val_data,
        output_dir=MODELS_DIR,
        epochs=20,
        batch_size=4,
        learning_rate=1e-5,
        max_length=512,
        gradient_accumulation=4,
        patience=5,
    )

    print(f"\n✅ Training complete!")
    print(f"   Best F1: {results['best_val_f1']:.4f} at epoch {results['best_epoch']}")
    print(f"   Model saved: {results['model_path']}")

    # Save test data (with all rich fields) for evaluation
    test_path = os.path.join(DATA_DIR, "test_data.jsonl")
    with open(test_path, 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"   Test data saved: {test_path}")

    return True


def step_evaluate():
    """Step 3: Evaluate the trained model using rich metrics."""
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING MODEL (DETAILED)")
    print("=" * 60)

    from ml_pipeline.evaluate import evaluate_on_testset, generate_evaluation_report

    test_path = os.path.join(DATA_DIR, "test_data.jsonl")

    if not os.path.exists(test_path):
        print("ERROR: test_data.jsonl not found. Train the model first.")
        return 0

    # Run detailed evaluation
    results = evaluate_on_testset(
        test_path=test_path,
        model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None,
        value_tolerance=0.15
    )

    # Generate report
    report_path = os.path.join(PIPELINE_DIR, "evaluation_report.txt")
    generate_evaluation_report(results, report_path)

    # Print summary
    overall = results['overall']
    print(f"\n📊 OVERALL METRICS:")
    print(f"  Precision:      {overall['precision']:.2%}")
    print(f"  Recall:         {overall['recall']:.2%}")
    print(f"  F1 Score:       {overall['f1']:.2%}")
    print(f"  Value Accuracy: {overall['value_accuracy']:.2%}")
    print(f"  Unit Accuracy:  {overall['unit_accuracy']:.2%}")
    print(f"  Exact Match:    {overall['exact_match']:.2%}")

    return overall['f1']


def step_analyze(pdf_path: str, industry: str = 'general'):
    """Analyze a single PDF report."""
    from ml_pipeline.pipeline import ESGMLPipeline

    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    pipeline = ESGMLPipeline(model_path=model_path, industry=industry)

    results = pipeline.analyze_report(pdf_path)

    if results:
        print(f"\n{'='*60}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"  Company: {results['company']}")
        print(f"  Industry: {results['industry']}")
        print(f"  Overall ESG Score: {results['esg_scores']['overall']['score']}/100")
        print(f"  Environmental: {results['esg_scores']['environmental']['score']}/100")
        print(f"  Social: {results['esg_scores']['social']['score']}/100")
        print(f"  Governance: {results['esg_scores']['governance']['score']}/100")
        print(f"\n  Metrics extracted: {len(results['extracted_metrics'])}")

        for m, info in sorted(results['extracted_metrics'].items()):
            print(f"    {m:25s}: {info['value']} {info['unit']} "
                  f"(conf: {info['confidence']:.2f})")

        print(f"\n  Recommendations: {len(results['recommendations'])}")
        for r in results['recommendations'][:5]:
            print(f"    [{r['priority']}] {r['metric']}: {r['recommendation']}")

        # Save results
        output_file = os.path.join(PIPELINE_DIR, "analysis_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_file}")

    return results


def step_serve(port: int = 5000):
    """Launch the Flask API server."""
    from ml_pipeline.api import create_app

    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    app = create_app(model_path=model_path)

    print(f"\n🚀 ESG ML API starting on http://localhost:{port}")
    print(f"   POST /api/analyze     - Upload PDF for analysis")
    print(f"   POST /api/analyze-text - Analyze raw text")
    print(f"   GET  /api/health      - Health check")
    app.run(host='0.0.0.0', port=port, debug=False)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == 'label':
        step_label()

    elif command == 'train':
        step_train()

    elif command == 'evaluate':
        f1 = step_evaluate()
        if f1 >= 0.85:
            print(f"\n🎉 Model F1 ({f1:.2%}) meets target (85%). Ready for deployment!")
        elif f1 >= 0.75:
            print(f"\n✓ Model F1 ({f1:.2%}) above baseline (75%). Consider more data for 85% target.")
        else:
            print(f"\n⚠ Model F1 ({f1:.2%}) below target (85%). Add more training data.")

    elif command == 'analyze':
        if len(sys.argv) < 3:
            print("Usage: python run_pipeline.py analyze <pdf_path> [industry]")
            return
        pdf_path = sys.argv[2]
        industry = sys.argv[3] if len(sys.argv) > 3 else 'general'
        step_analyze(pdf_path, industry)

    elif command == 'serve':
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        step_serve(port)

    elif command == 'all':
        print("🚀 RUNNING FULL PIPELINE (RoBERTa + Multi-Head)\n")
        if step_label():
            if step_train():
                f1 = step_evaluate()
                print(f"\n{'='*60}")
                if f1 >= 0.85:
                    print(f"🎉 Pipeline complete! F1: {f1:.2%} — Target met! Ready for deployment!")
                    print(f"   Run: python run_pipeline.py serve")
                elif f1 >= 0.75:
                    print(f"✓ Pipeline complete. F1: {f1:.2%} — Good progress toward 85% target.")
                    print(f"   Consider adding more PDFs to Dataset/ for further improvement.")
                else:
                    print(f"⚠ Pipeline complete. F1: {f1:.2%} — Below 85% target.")
                    print(f"   Add more labeled data to Dataset/ folder.")
                print(f"{'='*60}")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()