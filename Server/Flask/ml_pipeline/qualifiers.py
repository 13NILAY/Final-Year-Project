"""
Qualifier Extraction Module
============================
Extracts year, scope, actual/target, geography, entity level from text.
"""

import re
from typing import Dict, Optional

def extract_qualifiers(text: str, window: int = 100) -> Dict[str, Optional[str]]:
    """
    Extract qualifiers from a text window around a value.
    Returns a dictionary with keys: year, scope, actual_or_target, geography, entity_level.
    """
    qualifiers = {
        'year': None,
        'scope': None,
        'actual_or_target': 'actual',  # default
        'geography': 'global',
        'entity_level': 'company',
    }

    text_lower = text.lower()

    # --- Year extraction ---
    # Look for 4-digit year
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        qualifiers['year'] = year_match.group()
    else:
        # Fiscal year like FY2023, FY23, 2023-24
        fy_match = re.search(r'\bFY\s*(\d{2,4})\b', text, re.I)
        if fy_match:
            yr = fy_match.group(1)
            if len(yr) == 2:
                qualifiers['year'] = '20' + yr
            else:
                qualifiers['year'] = yr
        else:
            fy_range = re.search(r'\b(\d{4})[-–](\d{2})\b', text)
            if fy_range:
                qualifiers['year'] = fy_range.group(1)  # start year

    # --- Scope extraction ---
    scope_match = re.search(r'scope\s*([123])', text_lower)
    if scope_match:
        qualifiers['scope'] = int(scope_match.group(1))
    elif 'scope one' in text_lower or 'direct emission' in text_lower:
        qualifiers['scope'] = 1
    elif 'scope two' in text_lower or 'indirect emission' in text_lower:
        qualifiers['scope'] = 2
    elif 'scope three' in text_lower or 'value chain' in text_lower:
        qualifiers['scope'] = 3

    # --- Actual vs Target ---
    if re.search(r'\b(target|aim|goal|aspire|commitment)\b', text_lower):
        qualifiers['actual_or_target'] = 'target'
    elif re.search(r'\b(actual|reported|current|historical)\b', text_lower):
        qualifiers['actual_or_target'] = 'actual'
    elif re.search(r'\b(decrease|increase|reduction|change)\b.*\bby\b', text_lower):
        qualifiers['actual_or_target'] = 'change'

    # --- Geography ---
    geo_keywords = {
        'india': 'India',
        'global': 'global',
        'usa': 'USA',
        'us': 'USA',
        'europe': 'Europe',
        'uk': 'UK',
        'china': 'China',
    }
    for kw, geo in geo_keywords.items():
        if kw in text_lower:
            qualifiers['geography'] = geo
            break

    # --- Entity level ---
    if re.search(r'\b(subsidiary|plant|facility|site|division)\b', text_lower):
        qualifiers['entity_level'] = 'subsidiary'
    else:
        qualifiers['entity_level'] = 'company'

    return qualifiers