"""
Preprocessing Module
====================
PDF text extraction, cleaning, and chunking for ESG report processing.
Optimized chunk sizes for better ESG metric localization.
"""

import re
import warnings
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import pdfplumber
from pdf2image import convert_from_path
import easyocr
import numpy as np

# ─── INITIALISE EASYOCR READER (do it once to avoid reloading) ─────────────────
# You can add more languages if your reports contain them, e.g. ['en','hi']
_ocr_reader = easyocr.Reader(['en'], gpu=False)   # set gpu=True if you have a GPU
import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using a multi‑tier approach:
      1. PyMuPDF – fast, handles most digital PDFs.
      2. EasyOCR – deep‑learning OCR for scanned or damaged documents.
    Returns cleaned text, or empty string if all methods fail.
    """
    # ─── TIER 1: PyMuPDF ────────────────────────────────────────────────────
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text() + "\n\n"
        doc.close()
        if full_text.strip():
            print(f"  [Preprocessing] PyMuPDF succeeded: {len(full_text)} chars")
            return clean_text(full_text)
        else:
            print("  [Preprocessing] PyMuPDF gave empty text – trying OCR...")
    except Exception as e:
        print(f"  [Preprocessing] PyMuPDF error: {str(e)[:200]} – trying OCR...")

    # ─── TIER 2: EasyOCR (with pdf2image) ────────────────────────────────────
    try:
        # Convert PDF pages to images (use high DPI for better accuracy)
        images = convert_from_path(pdf_path, dpi=300)
        if not images:
            print("  [Preprocessing] pdf2image returned no images.")
            return ""

        full_text = ""
        for i, img in enumerate(images):
            # Convert PIL image to numpy array (EasyOCR expects numpy)
            img_np = np.array(img)
            # Perform OCR. detail=0 returns only text, paragraph=True groups text into paragraphs.
            result = _ocr_reader.readtext(img_np, detail=0, paragraph=True)
            page_text = "\n".join(result)
            full_text += page_text + "\n\n"
            print(f"    OCR page {i+1} – extracted {len(page_text)} chars")

        if full_text.strip():
            print(f"  [Preprocessing] EasyOCR succeeded: {len(full_text)} chars")
            return clean_text(full_text)
        else:
            print("  [Preprocessing] EasyOCR returned no text.")
    except Exception as e:
        print(f"  [Preprocessing] OCR failed: {str(e)[:200]}")

    # If all else fails
    return ""

def clean_text(raw_text: str) -> str:
    """
    Clean and normalize extracted PDF text.
    
    - Remove page numbers, headers/footers
    - Normalize whitespace and encoding issues
    - Remove non-printable characters
    - Preserve numeric values and units
    
    Args:
        raw_text: Raw text from PDF extraction
        
    Returns:
        Cleaned text string
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove common PDF artifacts
    # Page numbers like "Page 1 of 50", "- 1 -", "1 | Page"
    text = re.sub(r'(?i)page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'(?m)^\s*-?\s*\d+\s*-?\s*$', '', text)
    text = re.sub(r'\d+\s*\|\s*[Pp]age', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Fix common encoding issues
    text = text.replace('\u2019', "'")
    text = text.replace('\u2018', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = text.replace('\u200b', '')   # Zero-width space
    text = text.replace('\ufeff', '')   # BOM

    # Remove non-printable characters but keep newlines and tabs
    text = re.sub(r'[^\x20-\x7E\n\t,.\\%:;/()₂³°-]', ' ', text)

    # Normalize whitespace: collapse multiple spaces but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)    # Multiple spaces → single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines → double newline
    text = re.sub(r'(?m)^\s+$', '', text)   # Blank lines with whitespace

    # Fix number formatting issues from PDF extraction
    # "1, 234, 567" → "1,234,567" (Fix spaces in comma-separated numbers)
    text = re.sub(r'(\d),\s+(\d)', r'\1,\2', text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
    min_chunk_size: int = 30
) -> List[Dict]:
    """
    Split text into overlapping chunks for model input.
    
    Uses sentence-aware boundaries to avoid cutting mid-sentence.
    Each chunk includes metadata about its position in the document.
    
    Smaller chunks (256 words) improve ESG metric localization since
    most ESG metrics appear within 1-3 sentences.
    
    Args:
        text: Cleaned text to chunk
        chunk_size: Target number of words per chunk (default: 256)
        overlap: Number of overlapping words between consecutive chunks (default: 64)
        min_chunk_size: Minimum words for a valid chunk (default: 30)
        
    Returns:
        List of dicts with 'text', 'chunk_id', 'word_count'
    """
    if not text:
        return []

    # Check minimum word count before chunking
    word_count = len(text.split())
    if word_count < 200:
        print(f"  [Preprocessing] WARNING: Text has only {word_count} words (minimum 200 recommended)")
        if word_count < min_chunk_size:
            print(f"  [Preprocessing] Text too short for chunking, skipping")
            return []

    # Split into sentences using common sentence boundaries
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=\S)'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current_words = []
    chunk_id = 0

    for sentence in sentences:
        sentence_words = sentence.split()

        if not sentence_words:
            continue

        current_words.extend(sentence_words)

        # When we have enough words, create a chunk
        if len(current_words) >= chunk_size:
            chunk_text_str = ' '.join(current_words)
            chunks.append({
                'text': chunk_text_str,
                'chunk_id': chunk_id,
                'word_count': len(current_words),
            })
            chunk_id += 1

            # Keep overlap words for next chunk
            if overlap > 0 and overlap < len(current_words):
                current_words = current_words[-overlap:]
            else:
                current_words = []

    # Don't forget the last chunk
    if len(current_words) >= min_chunk_size:
        chunk_text_str = ' '.join(current_words)
        chunks.append({
            'text': chunk_text_str,
            'chunk_id': chunk_id,
            'word_count': len(current_words),
        })

    if chunks:
        avg_words = sum(c['word_count'] for c in chunks) // len(chunks)
        print(f"  [Preprocessing] Created {len(chunks)} text chunks "
              f"(avg {avg_words} words/chunk)")
    else:
        print(f"  [Preprocessing] Created 0 text chunks — text may be too short or cleaning removed content")
        print(f"  [Preprocessing] DEBUG: text length = {len(text)} chars, {word_count} words")

    return chunks


def preprocess_pdf(pdf_path: str, chunk_size: int = 256, overlap: int = 64) -> List[Dict]:
    """
    Complete preprocessing pipeline: PDF → clean text → chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Target words per chunk (default: 256)
        overlap: Overlap words between chunks (default: 64)
        
    Returns:
        List of text chunk dicts
    """
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return []

    cleaned = clean_text(raw_text)
    if not cleaned:
        return []

    chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
    return chunks


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        chunks = preprocess_pdf(pdf_path)
        print(f"\nTotal chunks: {len(chunks)}")
        if chunks:
            print(f"\nFirst chunk preview ({chunks[0]['word_count']} words):")
            print(chunks[0]['text'][:300] + "...")
    else:
        print("Usage: python preprocessing.py <path_to_pdf>")
