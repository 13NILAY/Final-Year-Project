"""
Flask API Module
================
REST API for ESG metric extraction from PDF reports.
Integrates with the existing Node.js backend.
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from .pipeline import ESGMLPipeline


def create_app(model_path: str = None, debug: bool = False) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        model_path: Path to trained model checkpoint
        debug: Enable debug mode
    """
    app = Flask(__name__)
    CORS(app)

    # Configure uploads
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'ecolens_uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

    # Initialize pipeline (lazy - created on first request)
    pipeline_instance = {}

    def get_pipeline(industry: str = 'general') -> ESGMLPipeline:
        key = industry
        if key not in pipeline_instance:
            pipeline_instance[key] = ESGMLPipeline(
                model_path=model_path,
                industry=industry,
            )
        return pipeline_instance[key]

    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        pipeline = get_pipeline()
        return jsonify({
            'status': 'healthy',
            'model_loaded': pipeline.extractor.model_loaded,
            'device': str(pipeline.extractor.device),
            'version': '1.0.0',
        })

    @app.route('/api/analyze', methods=['POST'])
    def analyze():
        """
        Analyze an uploaded ESG report PDF.
        
        Expects:
            - File: 'report' (PDF file)
            - Form field: 'industry' (optional, default: 'general')
            
        Returns:
            JSON with extracted metrics, scores, and recommendations
        """
        if 'report' not in request.files:
            return jsonify({'error': 'No report file uploaded'}), 400

        file = request.files['report']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400

        industry = request.form.get('industry', 'general')

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Analyze
            pipeline = get_pipeline(industry)
            results = pipeline.analyze_report(filepath)

            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

            if results is None:
                return jsonify({'error': 'Failed to analyze report. Check if PDF contains text.'}), 422

            # Ensure JSON serializable
            clean_results = _make_serializable(results)

            return jsonify({
                'success': True,
                'data': clean_results,
            })

        except Exception as e:
            # Clean up on error
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analyze-text', methods=['POST'])
    def analyze_text():
        """
        Analyze raw text for ESG metrics (for testing/debugging).
        
        Expects JSON body:
            {"text": "...", "industry": "general"}
        """
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        industry = data.get('industry', 'general')

        try:
            pipeline = get_pipeline(industry)
            metrics = pipeline.extractor.extract_from_text(text)

            # Calculate scores
            scores = pipeline._calculate_category_scores(metrics)

            return jsonify({
                'success': True,
                'data': {
                    'extracted_metrics': _make_serializable(metrics),
                    'esg_scores': _make_serializable(scores),
                    'total_metrics': len(metrics),
                }
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app


def _make_serializable(obj):
    """Convert numpy/torch types to JSON serializable Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # torch tensors
        return obj.item()
    else:
        return obj


if __name__ == '__main__':
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    app = create_app(model_path=model_path, debug=True)
    print(f"\n🚀 ESG ML API starting on http://localhost:{port}")
    print(f"   POST /api/analyze     - Upload PDF for analysis")
    print(f"   POST /api/analyze-text - Analyze raw text")
    print(f"   GET  /api/health      - Health check")
    app.run(host='0.0.0.0', port=port, debug=True)
