"""
Table-Aware Extraction Module
==============================
Extract ESG metrics from tables in PDFs using pdfplumber.
Handles BRSR-style tables, GRI content indexes, and common ESG data tables.
"""

import re
import pdfplumber
from typing import List, Dict, Optional, Any
from .canonical_metrics import get_alias_manager
from .qualifiers import extract_qualifiers

def extract_table_metrics(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract metric candidates from all tables in the PDF.
    Returns list of candidates with fields:
        text: combined row text for context
        value: extracted numeric value
        unit: inferred unit
        raw_match: the cell text containing the value
        qualifiers: dict with year, etc.
        page: page number
        source: 'table'
        row_data: full row for debugging
    """
    candidates = []
    alias_manager = get_alias_manager()

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                # Try to find year headers in first row
                headers = table[0]
                years = []
                for h in headers[1:]:  # skip first column (metric name)
                    if h and isinstance(h, str):
                        year_match = re.search(r'\b(19|20)\d{2}\b', h)
                        if year_match:
                            years.append(year_match.group())
                        else:
                            years.append(None)
                    else:
                        years.append(None)

                # Process data rows
                for row in table[1:]:
                    if not row or not row[0]:
                        continue
                    metric_phrase = str(row[0]).strip()
                    if not metric_phrase or len(metric_phrase) < 3:
                        continue

                    for col_idx, cell in enumerate(row[1:], start=1):
                        if cell is None:
                            continue
                        cell_str = str(cell).strip()
                        # Extract numeric value from cell
                        value = _extract_number_from_cell(cell_str)
                        if value is None:
                            continue

                        # Build context text for qualifier extraction
                        context = f"{metric_phrase} {cell_str}"
                        qualifiers = extract_qualifiers(context)
                        # Override year if we have a header year
                        if col_idx-1 < len(years) and years[col_idx-1]:
                            qualifiers['year'] = years[col_idx-1]

                        # Infer unit from metric phrase or cell
                        unit = _infer_unit(metric_phrase, cell_str)

                        candidates.append({
                            'text': context,
                            'value': value,
                            'unit': unit,
                            'raw_match': cell_str,
                            'qualifiers': qualifiers,
                            'page': page_num,
                            'source': 'table',
                            'row_data': {'metric_phrase': metric_phrase, 'row': row}
                        })
    return candidates

def _extract_number_from_cell(cell: str) -> Optional[float]:
    """Extract the first numeric value from a cell string."""
    # Remove commas, handle Indian format
    cleaned = re.sub(r'[^\d.,-]', '', cell)
    # Handle ranges like "45-50" – take first number?
    if '-' in cleaned:
        cleaned = cleaned.split('-')[0]
    # Find numbers
    match = re.search(r'[\d,]+\.?\d*', cleaned)
    if match:
        num_str = match.group().replace(',', '')
        try:
            return float(num_str)
        except ValueError:
            return None
    return None

def _infer_unit(metric_phrase: str, cell: str) -> str:
    """Infer unit from phrase or cell content."""
    text = (metric_phrase + ' ' + cell).lower()
    if any(kw in text for kw in ['tco2e', 't co2e', 'tonnes co2']):
        return 'tCO2e'
    if any(kw in text for kw in ['mwh', 'gwh', 'kwh']):
        return 'MWh'
    if any(kw in text for kw in ['m3', 'cubic meter', 'kilolitre']):
        return 'm3'
    if any(kw in text for kw in ['%', 'percent']):
        return '%'
    if any(kw in text for kw in ['crore', 'lakh', 'rs.', 'inr']):
        return 'INR Crore'
    if any(kw in text for kw in ['tonnes', 'metric tons']):
        return 'tonnes'
    return 'unknown'