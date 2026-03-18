"""
EcoLens ML Pipeline - ML-based ESG Metric Extraction
=====================================================
Hybrid RoBERTa + regex pipeline for extracting ESG metrics from PDF reports.
Multi-head architecture with Environmental/Social/Governance classifier heads.
"""

__version__ = "2.0.0"

from .preprocessing import extract_text_from_pdf, clean_text, chunk_text
from .extractor import ESGStagedExtractor
from .pipeline import ESGMLPipeline
from .canonical_metrics import get_alias_manager