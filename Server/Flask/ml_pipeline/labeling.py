"""
Labeling Module
===============
Generate training labels from existing regex-based extractor (weak supervision).
Creates labeled JSONL datasets from ESG PDF reports.
Includes data augmentation for improved model generalization.

IMPROVED: Expanded regex patterns, text normalization, scoring-based labeling,
and suspicious-chunk logging for higher recall (target: 434 → 1500+ positives).
"""

import os
import re
import json
import random
from typing import List, Dict, Tuple, Optional
from .preprocessing import preprocess_pdf, clean_text, extract_text_from_pdf


# ─── TEXT NORMALIZATION FOR PDF-BROKEN FORMATS ──────────────────────────────

_WORD_NUMBERS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90',
}

_MULTIPLIER_WORDS = {
    'hundred': 100, 'thousand': 1_000, 'lakh': 100_000,
    'million': 1_000_000, 'crore': 10_000_000, 'billion': 1_000_000_000,
}


def normalize_for_matching(text: str) -> str:
    """
    Normalize PDF-extracted text to improve regex matching.

    Fixes broken spacing, units, separators, multi-line breaks, and word-numbers.
    """
    if not text:
        return text

    result = text

    # 1. Collapse broken numbers: "45 , 000" → "45,000"
    result = re.sub(r'(\d)\s*,\s+(\d)', r'\1,\2', result)
    result = re.sub(r'(\d)\s+,\s*(\d)', r'\1,\2', result)

    # 2. Collapse broken decimals: "3 . 14" → "3.14"
    result = re.sub(r'(\d)\s*\.\s+(\d)', r'\1.\2', result)

    # 3. Normalize broken units
    result = re.sub(r'\bt\s+CO\s*2\s*e\b', 'tCO2e', result, flags=re.IGNORECASE)
    result = re.sub(r'\bt\s*CO\s*2\s*e\b', 'tCO2e', result, flags=re.IGNORECASE)
    result = re.sub(r'\b([MmGgKk])\s*[Ww]\s*[Hh]\b', lambda m: m.group(1).upper() + 'Wh', result)
    result = re.sub(r'\b([GgTt])\s+[Jj]\b', lambda m: m.group(1).upper() + 'J', result)

    # 4. Normalize separators: en-dash, em-dash, equals → colon
    result = result.replace('\u2013', ':').replace('\u2014', ':').replace('=', ':')

    # 5. Merge keyword line + value line: "Scope 1\n45,000" → "Scope 1 : 45,000"
    result = re.sub(
        r'((?:scope|emission|energy|water|waste|turnover|training|board|female|ethic|'
        r'renewable|ghg|carbon|co2|hazardous|whistleblower|csr|community|satisfaction|'
        r'lost\s+time|injury|independence|director|pay\s+ratio|compliance|recycl)'
        r'[^\n]{0,40}?)\s*\n\s*(\d)',
        r'\1 : \2',
        result,
        flags=re.IGNORECASE,
    )

    # 6. Convert word-numbers: "forty-two" → "42"
    def _word_to_num(match):
        word = match.group(0).lower().strip()
        parts = re.split(r'[-\s]+', word)
        total = 0
        for p in parts:
            if p in _WORD_NUMBERS:
                total += int(_WORD_NUMBERS[p])
        return str(total) if total > 0 else match.group(0)

    word_num_pattern = (
        r'\b(?:' + '|'.join(_WORD_NUMBERS.keys()) + r')'
        r'(?:[-\s]+(?:' + '|'.join(_WORD_NUMBERS.keys()) + r'))*\b'
    )
    result = re.sub(word_num_pattern, _word_to_num, result, flags=re.IGNORECASE)

    # 7. "per cent" → "percent"
    result = re.sub(r'\bper\s+cent\b', 'percent', result, flags=re.IGNORECASE)

    # 8. Collapse pipe whitespace: " | " → "|"
    result = re.sub(r'\s*\|\s*', '|', result)

    # 9. Indian comma format: "1,23,456" → "123456"
    result = re.sub(r'(\d),(\d{2}),(\d{3})\b', r'\1\2\3', result)
    result = re.sub(r'(\d{1,2}),(\d{2}),(\d{2}),(\d{3})\b', r'\1\2\3\4', result)

    # 10. BRSR parenthetical unit descriptions → short form
    result = re.sub(r'\((?:in\s+)?metric\s+tonn?e?s?\s+(?:of\s+)?co2\s*(?:eq(?:uivalent)?)?\)', '(tCO2e)', result, flags=re.IGNORECASE)
    result = re.sub(r'\((?:in\s+)?(?:kilo)?lit(?:er|re)s?\)', '(KL)', result, flags=re.IGNORECASE)
    result = re.sub(r'\((?:in\s+)?(?:metric\s+)?tonn?e?s?\)', '(MT)', result, flags=re.IGNORECASE)
    result = re.sub(r'\((?:in\s+)?(?:M|G|k)Wh\)', lambda m: f'({m.group(0).strip("()")})', result)
    result = re.sub(r'\((?:in\s+)?(?:GJ|TJ|PJ)\)', lambda m: f'({m.group(0).strip("()")})', result)
    result = re.sub(r'\((?:in\s+)?(?:cubic\s+met(?:er|re)s?|m3|m³)\)', '(m3)', result, flags=re.IGNORECASE)

    # 11. Rupee symbol normalization
    result = re.sub(r'[₹`]\s*', 'Rs. ', result)
    result = re.sub(r'\bRs\.?\s*', 'Rs. ', result, flags=re.IGNORECASE)
    result = re.sub(r'\bINR\s+', 'Rs. ', result, flags=re.IGNORECASE)

    # 12. FY prefix normalization: "FY 2023-24" → "FY2024", "FY23" stays
    result = re.sub(r'\bFY\s*(\d{4})\s*[-–]\s*(\d{2})\b', r'FY\2', result)
    result = re.sub(r'\bFY\s+(\d{2,4})\b', r'FY\1', result)

    return result


# ─── UNIT PATTERN BUILDING BLOCKS ──────────────────────────────────────────

_NUM = r'[\d,]+\.?\d*'
_EMISSIONS_UNITS = (
    r'(?:tco2e?|(?:metric\s+)?ton(?:ne)?s?(?:\s+(?:of\s+)?co2(?:\s*(?:eq|equivalent))?)?'
    r'|mt(?:co2e?)?|ktco2e?|gtco2e?|kg\s*co2e?)'
)
_ENERGY_UNITS = r'(?:mwh|gwh|kwh|twh|gj|tj|pj|gigajoules?|terajoules?|petajoules?|mtoe|toe|btu|mmbtu)'
_WATER_UNITS = r'(?:m3|m\u00b3|cubic\s+met(?:er|re)s?|(?:mega|kilo)?lit(?:er|re)s?|ml|kl|gallons?|acre[- ]?feet)'
_MASS_UNITS = r'(?:ton(?:ne)?s?|mt|kg|kilograms?|metric\s+ton(?:ne)?s?)'
_PCT = r'(?:%|percent|per\s*cent)'
_SEP = r'[\s:|\-\u2013\u2014=]*'


# ─── ESG METRIC DEFINITIONS ─────────────────────────────────────────────────

ESG_METRICS = {
    # ═══════════════════════ ENVIRONMENTAL ═══════════════════════
    'ghg_emissions': {
        'category': 'environmental',
        'patterns': [
            r'(?:total\s+)?(?:greenhouse\s+gas|ghg|carbon)\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\s+(?:of\s+)?(?:greenhouse\s+gas|ghg|carbon)\s+emissions?',
            r'(?:greenhouse\s+gas|ghg|carbon)\s+emissions?\s+(?:stood\s+at|were|amounted\s+to|totall?(?:ed|ing)|of|was|reached|recorded)\s+(?:approximately\s+|about\s+|around\s+|nearly\s+)?(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'(?:greenhouse\s+gas|ghg|carbon)\s+emissions?\s*\|(' + _NUM + r')\|?' + _EMISSIONS_UNITS + r'?',
            r'(' + _NUM + r')\s*(?:thousand|million|billion|lakh|crore)?\s*' + _EMISSIONS_UNITS + r'\s+(?:of\s+)?(?:greenhouse|ghg|total\s+carbon)',
            r'total\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'ghg\s*(?:inventory|footprint)\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'?',
            r'our\s+(?:total\s+)?(?:greenhouse\s+gas|ghg|carbon)\s+emissions?\s+(?:were|are|is)\s+(?:approximately\s+)?(' + _NUM + r')',
            # BRSR table: GHG emissions (tCO2e)|value
            r'(?:total\s+)?(?:ghg|greenhouse|carbon)\s+emissions?\s*\([^)]*\)\s*\|\s*(' + _NUM + r')',
            # Parenthetical: emissions (45,000 tCO2e)
            r'(?:total\s+)?(?:ghg|greenhouse|carbon)\s+emissions?\s*\((' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\)',
            # Avoided/cumulative
            r'(?:avoided|offset|reduced|saved)\s+(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            # FY format
            r'(?:total\s+)?(?:ghg|carbon)\s+emissions?.*?fy\d{2,4}\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
        ],
        'keywords': ['greenhouse gas', 'ghg', 'total emissions', 'carbon emissions', 'ghg inventory',
                     'carbon footprint', 'ghg footprint', 'total ghg', 'gas emission',
                     'total co2', 'carbon neutral', 'net zero', 'decarboni'],
        'unit': 'tCO2e'
    },
    'scope1_emissions': {
        'category': 'environmental',
        'patterns': [
            r'scope[\s\-]*1\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'scope[\s\-]*1\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'direct\s+(?:ghg\s+)?emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'scope\s*one\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\s+(?:of\s+|for\s+)?scope[\s\-]*1',
            r'scope[\s\-]*1[^|\n]{0,20}\|(' + _NUM + r')',
            r's1\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'scope[\s\-]*1\s+emissions?\s+(?:stood\s+at|were|was|amounted\s+to|totall?(?:ed|ing)|reached|recorded)\s+(?:approximately\s+|about\s+|around\s+)?(' + _NUM + r')',
            r'scope[\s\-]*1\s+emissions?\s*' + _SEP + r'(' + _NUM + r')(?:\s|$|,)',
            # BRSR: Scope 1 emissions (Metric tonnes of CO2 equivalent)|value
            r'scope[\s\-]*1\s+emissions?\s*\([^)]*\)\s*\|\s*(' + _NUM + r')',
            r'scope[\s\-]*1\s+emissions?\s*\((' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\)',
            r'scope[\s\-]*1\s+(?:and\s+scope\s*2\s+)?(?:ghg\s+)?emissions?\s+(?:were|was|totaled?)\s+(?:approximately\s+)?(' + _NUM + r')',
        ],
        'keywords': ['scope 1', 'scope one', 'direct emissions', 'scope-1', 's1 emissions',
                     'direct ghg', 'scope i ', 'scope 1 emission', 'direct emission'],
        'unit': 'tCO2e'
    },
    'scope2_emissions': {
        'category': 'environmental',
        'patterns': [
            r'scope[\s\-]*2\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'scope[\s\-]*2\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'indirect\s+(?:ghg\s+)?emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'scope\s*two\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\s+(?:of\s+|for\s+)?scope[\s\-]*2',
            r'scope[\s\-]*2[^|\n]{0,20}\|(' + _NUM + r')',
            r's2\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'scope[\s\-]*2\s+emissions?\s+(?:stood\s+at|were|was|amounted\s+to|totall?(?:ed|ing)|reached|recorded)\s+(?:approximately\s+|about\s+)?(' + _NUM + r')',
            r'(?:purchased|bought)\s+electricity\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'scope[\s\-]*2\s+emissions?\s*' + _SEP + r'(' + _NUM + r')(?:\s|$|,)',
            r'scope[\s\-]*2\s+emissions?\s*\([^)]*\)\s*\|\s*(' + _NUM + r')',
            r'scope[\s\-]*2\s+emissions?\s*\((' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\)',
        ],
        'keywords': ['scope 2', 'scope two', 'indirect emissions', 'scope-2', 's2 emissions',
                     'purchased electricity', 'scope ii', 'scope 2 emission', 'indirect ghg',
                     'market-based', 'location-based', 'energy indirect'],
        'unit': 'tCO2e'
    },
    'scope3_emissions': {
        'category': 'environmental',
        'patterns': [
            r'scope[\s\-]*3\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'scope[\s\-]*3\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'value\s*chain\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'other\s+indirect\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'scope\s*three\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\s+(?:of\s+|for\s+)?scope[\s\-]*3',
            r'scope[\s\-]*3[^|\n]{0,20}\|(' + _NUM + r')',
            r's3\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'scope[\s\-]*3\s+emissions?\s+(?:stood\s+at|were|was|amounted\s+to|totall?(?:ed|ing)|reached|recorded)\s+(?:approximately\s+|about\s+)?(' + _NUM + r')',
            r'(?:upstream|downstream)\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + '?',
            r'scope[\s\-]*3\s+emissions?\s*' + _SEP + r'(' + _NUM + r')(?:\s|$|,)',
            r'scope[\s\-]*3\s+emissions?\s*\([^)]*\)\s*\|\s*(' + _NUM + r')',
            r'scope[\s\-]*3\s+emissions?\s*\((' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\)',
        ],
        'keywords': ['scope 3', 'scope three', 'value chain', 'scope-3', 's3 emissions',
                     'other indirect', 'scope iii', 'upstream emission', 'downstream emission',
                     'supply chain emission', 'scope 3 emission', 'category 1', 'category 3'],
        'unit': 'tCO2e'
    },
    'co2_emissions': {
        'category': 'environmental',
        'patterns': [
            r'co2\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'carbon\s+dioxide\s+emissions?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'co2e?\s*' + _SEP + r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'carbon\s+dioxide.*?(' + _NUM + r')\s*' + _EMISSIONS_UNITS,
            r'(' + _NUM + r')\s*' + _EMISSIONS_UNITS + r'\s+(?:of\s+)?co2',
            r'co2\s+emissions?\s+(?:stood\s+at|were|amounted\s+to)\s+(?:approximately\s+)?(' + _NUM + r')',
            r'co2[^|\n]{0,15}\|(' + _NUM + r')',
        ],
        'keywords': ['co2', 'carbon dioxide', 'co2e', 'carbon emission', 'carbon dioxide emission',
                     'co2 equivalent'],
        'unit': 'tCO2e'
    },
    'energy_consumption': {
        'category': 'environmental',
        'patterns': [
            r'(?:total\s+)?energy\s+(?:consumption|use|usage|consumed)\s*' + _SEP + r'(' + _NUM + r')\s*' + _ENERGY_UNITS,
            r'consumed\s+(' + _NUM + r')\s*' + _ENERGY_UNITS + r'\s+(?:of\s+)?energy',
            r'(' + _NUM + r')\s*' + _ENERGY_UNITS + r'\s+(?:of\s+)?(?:total\s+)?energy',
            r'energy\s+(?:consumption|use|usage)[^|\n]{0,15}\|(' + _NUM + r')',
            r'electricity\s+(?:consumption|use|usage)\s*' + _SEP + r'(' + _NUM + r')\s*' + _ENERGY_UNITS,
            r'energy\s+(?:consumption|use)\s+(?:was|were|is|stood\s+at|amounted\s+to|reached|recorded)\s+(?:approximately\s+)?(' + _NUM + r')\s*' + _ENERGY_UNITS,
            r'(?:power|fuel)\s+consumption\s*' + _SEP + r'(' + _NUM + r')\s*' + _ENERGY_UNITS,
            r'total\s+energy\s*' + _SEP + r'(' + _NUM + r')\s*' + _ENERGY_UNITS,
            # BRSR: Total electricity consumption (MWh)|value
            r'(?:total\s+)?(?:electricity|energy)\s+(?:consumption|use)\s*\([^)]*\)\s*\|\s*(' + _NUM + r')',
            r'(?:total\s+)?energy\s+(?:consumption|use)\s*\((' + _NUM + r')\s*' + _ENERGY_UNITS + r'\)',
            r'consumed\s+(?:approximately\s+|about\s+)?(' + _NUM + r')\s*' + _ENERGY_UNITS,
            # Parenthetical: Total energy (167,890 MWh)
            r'(?:total\s+)?energy\s*\((' + _NUM + r')\s*' + _ENERGY_UNITS + r'\)',
        ],
        'keywords': ['energy consumption', 'energy use', 'energy usage', 'total energy',
                     'electricity consumption', 'power consumption', 'fuel consumption',
                     'energy consumed', 'energy intensity', 'kwh', 'mwh', 'gwh', 'gigajoule',
                     'terajoule', 'total electricity', 'energy audit'],
        'unit': 'MWh'
    },
    'renewable_energy': {
        'category': 'environmental',
        'patterns': [
            r'renewable\s+energy\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:of\s+)?energy\s+(?:from|is)\s+renewable',
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:from\s+)?renewable',
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+renewable',
            r'renewables?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:green|clean|alternative)\s+energy\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:share\s+of\s+)?renewable\s+(?:energy\s+)?(?:share|mix|proportion)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s+percent\s+(?:of\s+)?(?:our\s+)?energy.*?renewable',
            r'renewable\s+energy[^|\n]{0,15}\|([\d,]*\.?\d+)',
            r'(?:solar|wind|hydro|biomass)\s+(?:energy\s+)?(?:share|contribution)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            # BRSR: Percentage of energy from renewable
            r'(?:percentage|share|proportion)\s+(?:of\s+)?(?:total\s+)?energy\s+(?:from\s+)?renewable[^|\n]{0,20}\|\s*([\d,]*\.?\d+)',
            r'(?:percentage|share)\s+(?:of\s+)?(?:total\s+)?energy\s+(?:from\s+)?renewable\s+(?:sources?)?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
        ],
        'keywords': ['renewable energy', 'clean energy', 'green energy', 'solar', 'wind energy',
                     'renewables', 'renewable sources', 'renewable power', 'green power',
                     'solar energy', 'wind power', 'hydropower', 'biomass', 'renewable share',
                     'alternative energy', 'clean power'],
        'unit': '%'
    },
    'water_withdrawal': {
        'category': 'environmental',
        'patterns': [
            r'water\s+(?:withdrawal|consumption|usage|use|intake)\s*' + _SEP + r'(' + _NUM + r')\s*' + _WATER_UNITS,
            r'(' + _NUM + r')\s*' + _WATER_UNITS + r'\s+(?:of\s+)?water',
            r'water\s+(?:withdrawn|consumed|used)\s+(?:was\s+|were\s+)?(?:approximately\s+|about\s+)?(' + _NUM + r')\s*' + _WATER_UNITS,
            r'water\s+(?:withdrawal|consumption)[^|\n]{0,15}\|(' + _NUM + r')',
            r'(?:fresh\s*)?water\s+(?:withdrawal|abstraction)\s*' + _SEP + r'(' + _NUM + r')\s*' + _WATER_UNITS,
            r'water.*?(' + _NUM + r')\s*(?:kl|kilolitres?|megalitres?)',
            # BRSR: Total water withdrawal (in kilolitres)|value
            r'(?:total\s+)?water\s+(?:withdrawal|consumption|discharge)\s*\([^)]*\)\s*\|\s*(' + _NUM + r')',
            r'water\s+(?:withdrawal|consumption)\s*\((' + _NUM + r')\s*' + _WATER_UNITS + r'\)',
            r'(' + _NUM + r')\s*(?:million\s+)?(?:litres?|liters?|gallons?)\s+(?:of\s+)?water',
            # million cubic meters / million litres
            r'(' + _NUM + r')\s*million\s+' + _WATER_UNITS,
            r'water.*?(?:totaled?|totalled?)\s+(?:approximately\s+)?(' + _NUM + r')\s*(?:million\s+)?' + _WATER_UNITS,
        ],
        'keywords': ['water withdrawal', 'water consumption', 'water usage', 'water use',
                     'freshwater', 'water intake', 'water abstraction', 'water withdrawn',
                     'groundwater', 'surface water', 'water footprint', 'water discharge',
                     'water recycled', 'water harvesting', 'water intensity', 'total water'],
        'unit': 'm3'
    },
    'waste_recycled': {
        'category': 'environmental',
        'patterns': [
            r'waste\s+recycled\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'recycling\s+rate\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:of\s+)?waste\s*(?:was\s+|were\s+)?(?:recycled|diverted|recovered)',
            r'waste\s+diversion\s+rate\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'recycled\s+(' + _NUM + r')\s*' + _PCT + r'\s+(?:of\s+)?waste',
            r'waste\s+(?:recovery|reclamation)\s+(?:rate\s*)?' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'waste\s+(?:recycled|diversion|recovery)[^|\n]{0,15}\|([\d,]*\.?\d+)',
            r'(?:landfill\s+)?diversion\s+rate\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
        ],
        'keywords': ['waste recycled', 'recycling rate', 'waste diversion', 'waste recovery',
                     'circular economy', 'landfill diversion', 'waste reclamation',
                     'zero waste', 'waste diverted', 'recycled waste', 'waste to landfill'],
        'unit': '%'
    },
    'hazardous_waste': {
        'category': 'environmental',
        'patterns': [
            r'hazardous\s+waste\s*' + _SEP + r'(' + _NUM + r')\s*' + _MASS_UNITS,
            r'(' + _NUM + r')\s*' + _MASS_UNITS + r'\s+(?:of\s+)?hazardous\s+waste',
            r'(?:toxic|dangerous)\s+waste\s*' + _SEP + r'(' + _NUM + r')\s*' + _MASS_UNITS,
            r'hazardous\s+waste[^|\n]{0,15}\|(' + _NUM + r')',
            r'hazardous\s+waste\s+(?:generated|produced|disposed)\s*' + _SEP + r'(' + _NUM + r')\s*' + _MASS_UNITS,
            r'(?:e-?waste|electronic\s+waste)\s*' + _SEP + r'(' + _NUM + r')\s*' + _MASS_UNITS,
            # BRSR: hazardous waste (in MT)|value
            r'hazardous\s+waste\s*\([^)]*\)\s*\|\s*(' + _NUM + r')',
            r'hazardous\s+waste.*?(' + _NUM + r')\s*(?:mt\b|metric)',
        ],
        'keywords': ['hazardous waste', 'toxic waste', 'dangerous waste', 'e-waste',
                     'electronic waste', 'biomedical waste', 'chemical waste'],
        'unit': 'tonnes'
    },

    # ═══════════════════════ SOCIAL ═══════════════════════
    'employee_turnover': {
        'category': 'social',
        'patterns': [
            r'(?:employee\s+)?turnover\s+rate\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'attrition\s+rate\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'employee\s+(?:attrition|turnover)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:employee\s+)?(?:turnover|attrition)',
            r'(?:staff|workforce)\s+turnover\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:voluntary|involuntary)\s+(?:turnover|attrition)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'turnover[^|\n]{0,15}\|([\d,]*\.?\d+)',
        ],
        'keywords': ['employee turnover', 'turnover rate', 'attrition rate', 'voluntary turnover',
                     'staff turnover', 'workforce turnover', 'employee attrition',
                     'involuntary turnover', 'separation rate'],
        'unit': '%'
    },
    'female_representation': {
        'category': 'social',
        'patterns': [
            r'female\s+(?:representation|employees?|workforce|staff)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'women\s+(?:in\s+)?(?:workforce|employees?|staff)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'gender\s*(?:diversity|ratio)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:of\s+)?(?:workforce|employees?|staff)\s+(?:are|were|is)\s+(?:female|women)',
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:female|women)\s+(?:employees?|workforce|representation)',
            r'women\s+(?:representation|participation)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:female|women)[^|\n]{0,20}\|([\d,]*\.?\d+)',
            r'diversity.*?(?:female|women)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
        ],
        'keywords': ['female representation', 'women workforce', 'gender diversity',
                     'women employees', 'female employees', 'women in workforce',
                     'gender ratio', 'female participation', 'women representation',
                     'diversity and inclusion', 'women staff'],
        'unit': '%'
    },
    'training_hours': {
        'category': 'social',
        'patterns': [
            r'(?:average\s+)?training\s+hours?\s*(?:per\s+employee\s*)?' + _SEP + r'([\d,]+\.?\d*)\s*(?:hours?)?',
            r'training\s+hours?\s+per\s+employee\s*' + _SEP + r'([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*hours?\s+(?:of\s+)?training',
            r'training.*?([\d,]+\.?\d*)\s*hours?\s*(?:per\s+)?employee',
            r'learning\s+hours?\s*' + _SEP + r'([\d,]+\.?\d*)',
            r'(?:skill|capability)\s+development\s+hours?\s*' + _SEP + r'([\d,]+\.?\d*)',
            r'training[^|\n]{0,15}\|([\d,]+\.?\d*)',
            r'hours?\s+(?:of\s+)?training\s+per\s+employee\s*' + _SEP + r'([\d,]+\.?\d*)',
        ],
        'keywords': ['training hours', 'learning hours', 'skill development',
                     'training per employee', 'capability development', 'learning and development',
                     'l&d hours', 'development hours', 'hours of training'],
        'unit': 'hours'
    },
    'lost_time_injury': {
        'category': 'social',
        'patterns': [
            r'lost\s+time\s+injury\s+(?:frequency\s+)?rate\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'ltifr?\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'(?:injury|incident)\s+(?:frequency\s+)?rate\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'lost\s+time\s+injury.*?rate\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'trir\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'(?:reportable|recordable)\s+(?:injury|incident)\s+rate\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'(?:safety|accident)\s+(?:frequency\s+)?rate\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'(?:ltifr|trir|injury\s+rate)[^|\n]{0,15}\|([\d,]*\.?\d+)',
            # NEW: BRSR fatality/injury count + rate patterns
            r'(?:number\s+of\s+)?(?:fatalities?|deaths?)\s*\|\s*(\d+)',
            r'(?:number\s+of\s+)?lost\s+time\s+injur(?:y|ies)\s*\|\s*(\d+)',
            r'(?:number\s+of\s+)?(?:reportable|recordable)\s+(?:injur(?:y|ies)|incident)\s*\|\s*(\d+)',
            r'lost\s+time\s+injur(?:y|ies)\s*' + _SEP + r'(\d+)',
            r'(?:fatality|fatalities)\s*' + _SEP + r'(\d+)',
            r'(?:occupational\s+)?(?:health\s+and\s+)?safety.*?(?:rate|frequency)\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'(?:oifr|oir)\s*' + _SEP + r'([\d,]*\.?\d+)',
            # TRIR/LTIFR narrative: "was 0.54"
            r'(?:trir|ltifr?)\s+(?:was|were|stood\s+at|of)\s+(?:approximately\s+)?([\d,]*\.?\d+)',
        ],
        'keywords': ['lost time injury', 'ltir', 'ltifr', 'injury rate', 'safety rate',
                     'workplace injury', 'trir', 'recordable injury', 'incident rate',
                     'accident rate', 'safety performance', 'zero harm', 'fatality',
                     'fatalities', 'occupational safety', 'oifr', 'health and safety',
                     'man-days lost', 'injury frequency', 'near miss'],
        'unit': 'rate'
    },
    'employee_satisfaction': {
        'category': 'social',
        'patterns': [
            r'employee\s+satisfaction\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:employee\s+)?engagement\s+score\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:employee\s+)?satisfaction\s+(?:rate|score|index)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:employee\s+)?(?:satisfaction|engagement)',
            r'(?:gallup|pulse|enps|engagement)\s+(?:survey\s+)?score\s*' + _SEP + r'([\d,]*\.?\d+)',
            r'employee\s+(?:satisfaction|engagement)[^|\n]{0,15}\|([\d,]*\.?\d+)',
        ],
        'keywords': ['employee satisfaction', 'engagement score', 'employee engagement',
                     'satisfaction rate', 'enps', 'gallup', 'pulse survey',
                     'great place to work', 'employee morale', 'engagement index'],
        'unit': '%'
    },
    'community_investment': {
        'category': 'social',
        'patterns': [
            r'community\s+investment\s*' + _SEP + r'(' + _NUM + r')\s*(?:crore|million|billion|lakh|inr|usd|\$|\u20b9)',
            r'csr\s+(?:spend|expenditure|investment|outlay)\s*' + _SEP + r'(' + _NUM + r')\s*(?:crore|million|billion|lakh|inr|usd|\$|\u20b9)',
            r'(?:charitable|philanthropic|social)\s+(?:spend|giving|contribution|investment)\s*' + _SEP + r'(' + _NUM + r')\s*(?:crore|million|billion|lakh)',
            r'(?:\u20b9|rs\.?|inr)\s*(' + _NUM + r')\s*(?:crore|lakh|million)?\s*(?:towards|for|on)\s+(?:csr|community|social)',
            r'community\s+investment[^|\n]{0,15}\|(' + _NUM + r')',
            r'csr\s+budget\s*' + _SEP + r'(' + _NUM + r')\s*(?:crore|million|lakh)',
            # NEW: broader CSR/community patterns
            r'rs\.?\s*(' + _NUM + r')\s*(?:crore|lakh)\s+(?:towards|for|on|in)\s+(?:csr|community|social)',
            r'csr\s+(?:spend|expenditure|contribution)\s+(?:was|were|amounted\s+to|of)\s+(?:rs\.?\s*)?(' + _NUM + r')\s*(?:crore|lakh|million)',
            r'csr\s+(?:spend|obligation)\s+(?:as\s+)?(?:%|percent).*?rs\.?\s*(' + _NUM + r')\s*(?:crore|lakh)',
            r'(' + _NUM + r')\s*(?:crore|million|lakh)\s+(?:towards|for|on)\s+(?:csr|community|social)',
            r'community\s+(?:development|investment)\s+(?:was|were|amounted\s+to)\s+(?:approximately\s+)?(?:rs\.?\s*|\$\s*)?(' + _NUM + r')',
            r'(?:social|community)\s+(?:investment|spend)\s*\|\s*(?:rs\.?\s*)?(' + _NUM + r')',
        ],
        'keywords': ['community investment', 'csr spend', 'social investment', 'charitable',
                     'philanthropy', 'csr expenditure', 'community development',
                     'social responsibility', 'csr budget', 'social spending',
                     'csr contribution', 'csr obligation', 'csr outlay',
                     'section 135', 'schedule vii', 'social impact'],
        'unit': 'INR Crore'
    },

    # ═══════════════════════ GOVERNANCE ═══════════════════════
    'board_independence': {
        'category': 'governance',
        'patterns': [
            r'(?:board\s+)?independence?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'independent\s+directors?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:of\s+)?(?:board\s+)?(?:are\s+)?independent',
            r'non[-\s]executive\s+directors?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'board\s+independence[^|\n]{0,15}\|([\d,]*\.?\d+)',
            r'(?:proportion|percentage|share)\s+(?:of\s+)?independent\s+directors?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+board\s+independence',
        ],
        'keywords': ['board independence', 'independent directors', 'independent board',
                     'non-executive directors', 'board composition', 'independent members',
                     'outside directors', 'board diversity'],
        'unit': '%'
    },
    'female_directors': {
        'category': 'governance',
        'patterns': [
            r'female\s+directors?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'women\s+(?:on\s+)?(?:the\s+)?board\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'.*?women.*?board',
            r'board\s+gender\s+diversity\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'women\s+directors?\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:female|women)\s+(?:representation\s+(?:on\s+)?)?board\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:female|women)\s+directors?[^|\n]{0,15}\|([\d,]*\.?\d+)',
        ],
        'keywords': ['female directors', 'women on board', 'women directors', 'board gender diversity',
                     'female board members', 'women board members', 'board gender',
                     'gender diversity board'],
        'unit': '%'
    },
    'ceo_pay_ratio': {
        'category': 'governance',
        'patterns': [
            r'ceo\s+(?:pay|compensation|remuneration)\s+ratio\s*' + _SEP + r'([\d,]+\.?\d*)',
            r'(?:pay|compensation|remuneration)\s+ratio\s*' + _SEP + r'(\d+\.?\d*)\s*[:\s]?\s*1',
            r'executive\s+(?:pay|compensation)\s+ratio\s*' + _SEP + r'(\d+\.?\d*)\s*:?\s*1',
            r'md\s+(?:pay|compensation|remuneration)\s+ratio\s*' + _SEP + r'([\d,]+\.?\d*)',
            r'ceo\s+(?:to\s+)?(?:median\s+)?(?:worker\s+)?(?:pay\s+)?ratio\s*' + _SEP + r'([\d,]+\.?\d*)',
        ],
        'keywords': ['ceo pay ratio', 'compensation ratio', 'pay ratio', 'executive compensation',
                     'ceo remuneration', 'md compensation', 'pay equity ratio',
                     'ceo to worker', 'executive pay'],
        'unit': 'ratio'
    },
    'ethics_training': {
        'category': 'governance',
        'patterns': [
            r'ethics\s+training\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'ethics\s+training\s+completion\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'compliance\s+training\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'compliance\s+training\s+(?:completion|coverage)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'.*?(?:ethics|compliance)\s+training',
            r'code\s+of\s+conduct\s+training\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:anti[-\s]?corruption|anti[-\s]?bribery)\s+training\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:ethics|compliance|code\s+of\s+conduct)\s+training[^|\n]{0,15}\|([\d,]*\.?\d+)',
            r'(?:posh|sexual\s+harassment)\s+(?:awareness\s+)?training\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            # NEW: broader patterns for low-frequency metric
            r'(?:employees?|staff|workers?)\s+(?:covered|trained)\s+(?:under|in)\s+(?:ethics|compliance)\s+training\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:ethics|compliance|integrity)\s+(?:training|program)\s*\([^)]*\)\s*\|\s*([\d,]*\.?\d+)',
            r'(?:abms|iso\s*37001).*?([\d,]*\.?\d+)\s*' + _PCT,
            r'trained\s+on\s+(?:ethics|compliance|code\s+of\s+conduct|anti[- ]?bribery)\s*' + _SEP + r'([\d,]*\.?\d+)\s*' + _PCT,
            r'(?:ethics|compliance)\s+(?:training|awareness)\s+(?:covered|reached|provided\s+to)\s+([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+(?:of\s+)?(?:employees?|staff)\s+(?:completed?|covered|received)\s+(?:ethics|compliance)',
            # ISO 37001 / ABMS mention with percentage
            r'(?:abms|iso\s*37001)[^.]{0,80}?([\d,]*\.?\d+)\s*' + _PCT,
            r'([\d,]*\.?\d+)\s*' + _PCT + r'\s+compliance\s+training\s+coverage',
        ],
        'keywords': ['ethics training', 'compliance training', 'code of conduct',
                     'anti-corruption training', 'ethics training completion',
                     'anti-bribery', 'integrity training', 'posh training',
                     'code of conduct training', 'ethical conduct', 'abms',
                     'iso 37001', 'ethics program', 'integrity program',
                     'compliance program', 'ethics awareness'],
        'unit': '%'
    },
    'whistleblower_cases': {
        'category': 'governance',
        'patterns': [
            r'whistle\s*blower\s+(?:cases?|complaints?|reports?|incidents?)\s*' + _SEP + r'([\d,]+)',
            r'([\d,]+)\s*whistle\s*blower\s+(?:cases?|complaints?)',
            r'(?:grievance|vigil)\s+(?:mechanism\s+)?(?:cases?|complaints?|reports?)\s*' + _SEP + r'(\d+)',
            r'(?:ethics\s+)?(?:hotline|helpline)\s+(?:cases?|complaints?|calls?)\s*' + _SEP + r'(\d+)',
            r'complaints?\s+received\s*' + _SEP + r'(\d+)',
            r'(\d+)\s+(?:grievance|ethics|integrity)\s+(?:cases?|complaints?)',
            r'whistle\s*blower[^|\n]{0,15}\|(\d+)',
            r'vigil\s+mechanism\s*' + _SEP + r'(\d+)\s+(?:cases?|complaints?)',
            # NEW: BRSR-specific patterns
            r'(?:number\s+of\s+)?complaints?\s+(?:filed|reported|received)\s+during\s+the\s+year\s*\|\s*whistle\s*blower\s*\|\s*(\d+)',
            r'complaints?\s+(?:filed|reported|received).*?whistle\s*blower.*?(\d+)',
            r'(?:number\s+of\s+)?complaints?\s+(?:filed|reported)\s*' + _SEP + r'(\d+)',
            r'whistle\s*blower\s*\|\s*(\d+)',
            r'vigil\s+mechanism[^|\n]{0,30}\|(\d+)',
            r'(?:sexual\s+harassment|posh)\s+(?:cases?|complaints?)\s*' + _SEP + r'(\d+)',
        ],
        'keywords': ['whistleblower', 'grievance', 'complaints received', 'ethical concerns',
                     'vigil mechanism', 'ethics hotline', 'integrity helpline',
                     'speak up', 'ethics complaints', 'grievance mechanism',
                     'whistleblowing', 'complaints filed', 'posh complaints',
                     'sexual harassment', 'ombudsman'],
        'unit': 'count'
    },
}


# ─── DERIVED CONSTANTS ──────────────────────────────────────────────────────

CATEGORY_MAP = {
    'environmental': [k for k, v in ESG_METRICS.items() if v['category'] == 'environmental'],
    'social': [k for k, v in ESG_METRICS.items() if v['category'] == 'social'],
    'governance': [k for k, v in ESG_METRICS.items() if v['category'] == 'governance'],
}

METRIC_NAMES = list(ESG_METRICS.keys()) + ['no_metric']
METRIC_TO_ID = {name: idx for idx, name in enumerate(METRIC_NAMES)}
ID_TO_METRIC = {idx: name for name, idx in METRIC_TO_ID.items()}
NUM_CLASSES = len(METRIC_NAMES)

# ESG context words for scoring
_ESG_CONTEXT_WORDS = frozenset([
    'sustainability', 'esg', 'report', 'brsr', 'annual', 'disclosure',
    'environment', 'social', 'governance', 'metric', 'performance',
    'target', 'fy', 'fiscal', 'quarter', 'gri', 'sasb', 'tcfd',
])

# Sanity bounds per metric (reuse from extractor or define here)
_METRIC_VALUE_BOUNDS = {
    'ceo_pay_ratio': (1, 500),
    'lost_time_injury': (0, 50),
    'renewable_energy': (0, 100),
    'waste_recycled': (0, 100),
    'employee_turnover': (0, 100),
    'female_representation': (0, 100),
    'board_independence': (0, 100),
    'female_directors': (0, 100),
    'ethics_training': (0, 100),
    'employee_satisfaction': (0, 100),
    'ghg_emissions': (0, 1e12),
    'scope1_emissions': (0, 1e12),
    'scope2_emissions': (0, 1e12),
    'scope3_emissions': (0, 1e12),
    'co2_emissions': (0, 1e12),
    'energy_consumption': (0, 1e12),
    'water_withdrawal': (0, 1e12),
    'hazardous_waste': (0, 1e12),
    'training_hours': (0, 1000),
    'whistleblower_cases': (0, 100000),
    'community_investment': (0, 1e12),
}


# ─── EXTRACTION & LABELING FUNCTIONS ────────────────────────────────────────

def _extract_value(text: str, patterns: List[str]) -> Optional[Tuple[float, str]]:
    """Try to extract a numeric value from text using patterns."""
    text_lower = text.lower()
    for pattern in patterns:
        try:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE))
        except re.error:
            continue
        for match in matches:
            try:
                value_str = match.group(1).replace(',', '').strip()
                if not value_str:
                    continue
                value = float(value_str)

                # Handle multipliers in the match text
                match_text = match.group(0).lower()
                if 'million' in match_text:
                    value *= 1_000_000
                elif 'billion' in match_text:
                    value *= 1_000_000_000
                elif 'thousand' in match_text or 'lakh' in match_text:
                    value *= 1_000
                elif 'crore' in match_text:
                    value *= 10_000_000

                return value, match.group(0)
            except (ValueError, IndexError):
                continue
    return None


def _has_metric_keywords(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the metric's keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _count_keywords(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _has_esg_context(text: str) -> bool:
    """Check if text contains ESG context words (sustainability report, BRSR, etc.)."""
    words = set(text.lower().split())
    return bool(words & _ESG_CONTEXT_WORDS)


def _is_table_format(text: str) -> bool:
    """Check if text appears to be from a table (pipes, consistent spacing)."""
    return '|' in text or bool(re.search(r'\d+\s{2,}\d+', text))


def _value_in_bounds(value: float, metric_name: str) -> bool:
    """Check if extracted value is within sanity bounds for the metric."""
    if metric_name in _METRIC_VALUE_BOUNDS:
        lo, hi = _METRIC_VALUE_BOUNDS[metric_name]
        return lo <= value <= hi
    return True


def label_chunk(chunk_text: str) -> List[Dict]:
    """
    Label a single text chunk using SCORING-BASED acceptance.

    Scoring system (replaces strict regex+keyword AND gate):
        +3  regex pattern matches with valid value
        +2  strong unit match in text (tCO2e, MWh, etc.)
        +2  keyword_count >= 2
        +1  keyword_count == 1
        +1  ESG context words present
        +1  table format detected

    Acceptance threshold: score >= 3 (was effectively >= 5 before)

    Returns list of label dicts found in the chunk.
    """
    # Normalize text for matching (fix broken PDF artifacts)
    normalized = normalize_for_matching(chunk_text)
    labels = []

    has_context = _has_esg_context(normalized)
    is_table = _is_table_format(normalized)

    for metric_name, metric_def in ESG_METRICS.items():
        result = _extract_value(normalized, metric_def['patterns'])
        if not result:
            continue

        value, raw_match = result
        score = 3  # Base score for regex match

        # Sanity check: value must be in bounds
        if not _value_in_bounds(value, metric_name):
            continue

        # Keyword scoring
        kw_count = _count_keywords(normalized, metric_def['keywords'])
        if kw_count >= 2:
            score += 2
        elif kw_count == 1:
            score += 1

        # Context bonuses
        if has_context:
            score += 1
        if is_table:
            score += 1

        # Check for strong unit match (adds confidence even without keywords)
        unit_patterns = {
            'tCO2e': r'tco2e?|tonnes?\s+co2',
            'MWh': r'mwh|gwh|kwh|gigajoule',
            'm3': r'm3|m\u00b3|cubic\s+met',
            '%': r'%|percent',
            'ratio': r':\s*1\b',
        }
        expected_unit = metric_def.get('unit', '')
        if expected_unit in unit_patterns:
            if re.search(unit_patterns[expected_unit], normalized.lower()):
                score += 2

        # Accept if score >= 3 (regex match alone is enough with unit)
        if score >= 3:
            labels.append({
                'metric_name': metric_name,
                'category': metric_def['category'],
                'value': value,
                'unit': metric_def['unit'],
                'raw_match': raw_match,
                'score': score,
            })

    return labels


def label_chunk_keyword(chunk_text: str) -> Optional[str]:
    """
    Classify a chunk by keyword presence (for chunks without extractable values).
    IMPROVED: requires 1 keyword + ESG context, or 2+ keywords.
    Returns the most likely metric name or None.
    """
    normalized = normalize_for_matching(chunk_text)
    text_lower = normalized.lower()
    has_context = _has_esg_context(normalized)
    best_metric = None
    best_score = 0

    for metric_name, metric_def in ESG_METRICS.items():
        kw_score = sum(1 for kw in metric_def['keywords'] if kw in text_lower)
        if kw_score > best_score:
            best_score = kw_score
            best_metric = metric_name

    # Accept with 2+ keywords, or 1 keyword + ESG context
    if best_score >= 2:
        return best_metric
    if best_score >= 1 and has_context:
        return best_metric
    return None


def label_chunk_suspicious(chunk_text: str) -> Optional[Dict]:
    """
    Identify 'suspicious' chunks that nearly match ESG metrics but fall below
    the acceptance threshold. Used for logging near-misses for manual review.

    Returns dict with {suspected_metric, score, reason} or None.
    """
    normalized = normalize_for_matching(chunk_text)
    text_lower = normalized.lower()

    best_result = None
    best_score = 0

    for metric_name, metric_def in ESG_METRICS.items():
        score = 0
        reasons = []

        # Check keywords
        kw_count = sum(1 for kw in metric_def['keywords'] if kw in text_lower)
        if kw_count >= 1:
            score += kw_count
            reasons.append(f'{kw_count} keyword(s)')

        # Check for numbers near keywords
        if re.search(r'\d{2,}', text_lower) and kw_count >= 1:
            score += 1
            reasons.append('has_numbers')

        # Check ESG context
        if _has_esg_context(normalized):
            score += 1
            reasons.append('esg_context')

        # Score 1-4 is "suspicious but not positively labeled"
        if 1 <= score <= 4 and score > best_score:
            best_score = score
            best_result = {
                'suspected_metric': metric_name,
                'score': score,
                'reason': ', '.join(reasons),
            }

    return best_result


# Low-frequency metrics get lower acceptance threshold
_LOW_FREQ_METRICS = frozenset([
    'ethics_training', 'whistleblower_cases', 'lost_time_injury',
    'community_investment', 'hazardous_waste', 'employee_satisfaction',
    'waste_recycled', 'ceo_pay_ratio',
])

# Unit → metric mapping for near-miss mining
_UNIT_METRIC_MAP = {
    'tco2e': ['ghg_emissions', 'scope1_emissions', 'scope2_emissions', 'scope3_emissions', 'co2_emissions'],
    'tco2': ['ghg_emissions', 'scope1_emissions', 'scope2_emissions', 'scope3_emissions', 'co2_emissions'],
    'mwh': ['energy_consumption'], 'gwh': ['energy_consumption'], 'kwh': ['energy_consumption'],
    'gj': ['energy_consumption'], 'tj': ['energy_consumption'],
    'kl': ['water_withdrawal'], 'kilolitre': ['water_withdrawal'], 'm3': ['water_withdrawal'],
    'crore': ['community_investment'], 'lakh': ['community_investment'],
}


def mine_suspicious_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Convert high-confidence suspicious chunks into labeled training samples.

    Algorithm:
    1. For each unlabeled chunk with 2+ ESG keywords + a number:
       a. Check if a recognized ESG unit appears → map to metric
       b. Check if 3+ keywords match a single metric → assign it
       c. If keyword+unit agree on same metric → label with confidence='mined'
    
    Returns list of mined sample dicts (same format as label_chunk output).
    """
    mined = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        if not text:
            continue
        
        normalized = normalize_for_matching(text)
        text_lower = normalized.lower()
        
        # Skip if already has a strong label
        labels = label_chunk(text)
        if labels:
            continue
        
        # Must have a number
        if not re.search(r'\d{2,}', text_lower):
            continue
        
        # Find best metric by keyword score
        best_metric = None
        best_kw_score = 0
        for metric_name, metric_def in ESG_METRICS.items():
            kw_count = sum(1 for kw in metric_def['keywords'] if kw in text_lower)
            if kw_count > best_kw_score:
                best_kw_score = kw_count
                best_metric = metric_name
        
        if best_kw_score < 2:
            continue
        
        # Check for unit presence that confirms the metric
        unit_confirmed = False
        for unit_key, metric_list in _UNIT_METRIC_MAP.items():
            if unit_key in text_lower and best_metric in metric_list:
                unit_confirmed = True
                break
        
        # Also check for % in percentage metrics
        pct_metrics = {'renewable_energy', 'waste_recycled', 'employee_turnover',
                       'female_representation', 'board_independence', 'female_directors',
                       'ethics_training', 'employee_satisfaction'}
        if best_metric in pct_metrics and ('%' in text or 'percent' in text_lower):
            unit_confirmed = True
        
        # Accept if: (3+ keywords) OR (2+ keywords + unit confirmed)
        if best_kw_score >= 3 or (best_kw_score >= 2 and unit_confirmed):
            # Try to extract a value
            metric_def = ESG_METRICS[best_metric]
            result = _extract_value(normalized, metric_def['patterns'])
            
            if result:
                value, raw_match = result
                if _value_in_bounds(value, best_metric):
                    mined.append({
                        'metric_name': best_metric,
                        'category': metric_def['category'],
                        'value': value,
                        'unit': metric_def['unit'],
                        'raw_match': raw_match,
                        'source': 'mined',
                    })
            else:
                # Even without value extraction, label for classification training
                mined.append({
                    'metric_name': best_metric,
                    'category': metric_def['category'],
                    'value': None,
                    'unit': metric_def['unit'],
                    'raw_match': None,
                    'source': 'mined_keyword_only',
                })
    
    return mined


# ─── DATASET GENERATION ─────────────────────────────────────────────────────

def generate_labeled_dataset(
    pdf_dir: str,
    output_path: str,
    chunk_size: int = 256,
    overlap: int = 64,
    include_negatives: bool = True,
    negative_ratio: float = 0.6,
) -> Dict:
    """
    Generate a labeled JSONL dataset from all PDFs in a directory structure.

    Expected directory structure:
        pdf_dir/
            Tech/
                report1.pdf
            Finance/
                ...

    Returns:
        Stats dict with counts
    """
    samples = []
    stats = {
        'total_pdfs': 0,
        'total_chunks': 0,
        'labeled_chunks': 0,
        'negative_chunks': 0,
        'suspicious_chunks': 0,
        'metrics_found': {},
        'by_sector': {},
        'skipped_pdfs': [],
    }

    sectors = [d for d in os.listdir(pdf_dir)
               if os.path.isdir(os.path.join(pdf_dir, d))]

    for sector in sectors:
        sector_path = os.path.join(pdf_dir, sector)
        pdf_files = [f for f in os.listdir(sector_path) if f.lower().endswith('.pdf')]
        stats['by_sector'][sector] = {'pdfs': len(pdf_files), 'samples': 0}

        print(f"\n[Labeling] Processing sector: {sector} ({len(pdf_files)} PDFs)")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(sector_path, pdf_file)
            stats['total_pdfs'] += 1

            try:
                chunks = preprocess_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    print(f"  [Labeling] WARNING: No chunks extracted from {pdf_file}")
                    stats['skipped_pdfs'].append({'file': pdf_file, 'reason': 'no_chunks'})
                    continue

                stats['total_chunks'] += len(chunks)

                positive_chunks = []
                negative_chunks = []

                for chunk in chunks:
                    labels = label_chunk(chunk['text'])

                    if labels:
                        for label in labels:
                            sample = {
                                'text': chunk['text'],
                                'metric_name': label['metric_name'],
                                'category': label['category'],
                                'value': label['value'],
                                'unit': label['unit'],
                                'sector': sector,
                                'source_pdf': pdf_file,
                                'chunk_id': chunk['chunk_id'],
                            }
                            positive_chunks.append(sample)
                            stats['labeled_chunks'] += 1

                            m = label['metric_name']
                            stats['metrics_found'][m] = stats['metrics_found'].get(m, 0) + 1
                    else:
                        # Try keyword-only classification for unlabeled chunks
                        kw_metric = label_chunk_keyword(chunk['text'])
                        if kw_metric:
                            sample = {
                                'text': chunk['text'],
                                'metric_name': kw_metric,
                                'category': ESG_METRICS[kw_metric]['category'],
                                'value': None,
                                'unit': ESG_METRICS[kw_metric]['unit'],
                                'sector': sector,
                                'source_pdf': pdf_file,
                                'chunk_id': chunk['chunk_id'],
                            }
                            positive_chunks.append(sample)
                            stats['labeled_chunks'] += 1
                            m = kw_metric
                            stats['metrics_found'][m] = stats['metrics_found'].get(m, 0) + 1
                        else:
                            suspicious = label_chunk_suspicious(chunk['text'])
                            if suspicious:
                                stats['suspicious_chunks'] += 1
                            negative_chunks.append(chunk)

                # Mine suspicious chunks for additional positives
                mined = mine_suspicious_chunks(negative_chunks)
                if mined:
                    for m_label in mined:
                        sample = {
                            'text': next((c['text'] for c in negative_chunks), ''),
                            'metric_name': m_label['metric_name'],
                            'category': m_label['category'],
                            'value': m_label['value'],
                            'unit': m_label['unit'],
                            'sector': sector,
                            'source_pdf': pdf_file,
                            'chunk_id': 'mined',
                        }
                        positive_chunks.append(sample)
                        stats['labeled_chunks'] += 1
                        mn = m_label['metric_name']
                        stats['metrics_found'][mn] = stats['metrics_found'].get(mn, 0) + 1
                    stats['mined_chunks'] = stats.get('mined_chunks', 0) + len(mined)

                samples.extend(positive_chunks)
                stats['by_sector'][sector]['samples'] += len(positive_chunks)

                # Add negative samples
                if include_negatives and negative_chunks:
                    n_negatives = max(1, int(len(positive_chunks) * negative_ratio))
                    selected_negatives = random.sample(
                        negative_chunks, min(n_negatives, len(negative_chunks))
                    )
                    for neg_chunk in selected_negatives:
                        sample = {
                            'text': neg_chunk['text'],
                            'metric_name': 'no_metric',
                            'category': 'none',
                            'value': None,
                            'unit': None,
                            'sector': sector,
                            'source_pdf': pdf_file,
                            'chunk_id': neg_chunk['chunk_id'],
                        }
                        samples.append(sample)
                        stats['negative_chunks'] += 1

            except Exception as e:
                print(f"  [Labeling] ERROR processing {pdf_file}: {str(e)[:200]}")
                stats['skipped_pdfs'].append({'file': pdf_file, 'reason': f'exception: {str(e)[:100]}'})
                continue

    # Augment positive samples
    print(f"\n[Labeling] Augmenting positive samples...")
    try:
        from .augmentation import augment_samples
        augmented = augment_samples(
            [s for s in samples if s['metric_name'] != 'no_metric'],
            multiplier=8,
        )
        samples.extend(augmented)
        print(f"  [Labeling] Added {len(augmented)} augmented samples")
    except ImportError:
        print(f"  [Labeling] WARNING: augmentation module not found, skipping")
    except Exception as e:
        print(f"  [Labeling] WARNING: augmentation failed: {str(e)[:100]}")

    random.shuffle(samples)

    # Save to JSONL
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n[Labeling] Dataset saved to {output_path}")
    print(f"  Total PDFs processed: {stats['total_pdfs']}")
    print(f"  Total chunks created: {stats['total_chunks']}")
    print(f"  Labeled (positive) samples: {stats['labeled_chunks']}")
    print(f"  Mined from near-misses: {stats.get('mined_chunks', 0)}")
    print(f"  Negative samples: {stats['negative_chunks']}")
    print(f"  Suspicious near-misses: {stats['suspicious_chunks']}")
    print(f"  Metric distribution: {stats['metrics_found']}")
    if stats['skipped_pdfs']:
        print(f"  Skipped PDFs: {len(stats['skipped_pdfs'])}")
        for skip in stats['skipped_pdfs'][:5]:
            print(f"    - {skip['file']}: {skip['reason']}")

    return stats


def load_dataset(path: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List, List, List]:
    """
    Load JSONL dataset and split into train/val/test.
    """
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]

    print(f"[Dataset] Loaded {n} samples -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


if __name__ == "__main__":
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = os.path.join(project_root, "Dataset")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "esg_labeled.jsonl")

    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print(f"Generating labeled dataset from: {dataset_dir}")
    print(f"Output: {output_file}")

    stats = generate_labeled_dataset(dataset_dir, output_file)

    train, val, test = load_dataset(output_file)
    if train:
        print(f"\nSample training example:")
        print(json.dumps(train[0], indent=2, ensure_ascii=False)[:500])
