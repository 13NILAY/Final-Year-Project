"""
Data Augmentation Module
========================
Generate augmented ESG training samples through synonym replacement,
number perturbation, template-based generation, and sentence shuffling.
Designed to increase training data diversity without requiring new PDFs.
"""

import re
import random
from typing import List, Dict, Optional
from copy import deepcopy


# ─── ESG SYNONYM DICTIONARIES ───────────────────────────────────────────────
def _add_pdf_noise(text: str) -> str:
    import random
    
    # Random line breaks
    words = text.split()
    for i in range(0, len(words), random.randint(8, 15)):
        if random.random() < 0.3:
            words[i] = words[i] + "\n"
    
    noisy = " ".join(words)
    
    # Add random spacing noise
    noisy = noisy.replace("  ", " ")
    noisy = noisy.replace(":", " : ")
    
    return noisy

ESG_SYNONYMS = {
    # Emission terms
    'emissions': ['discharge', 'output', 'release', 'generation'],
    'greenhouse gas': ['ghg', 'carbon', 'CO2 equivalent'],
    'scope 1': ['direct emissions', 'scope one', 'scope-1', 'S1'],
    'scope 2': ['indirect emissions', 'scope two', 'scope-2', 'S2', 'purchased electricity emissions'],
    'scope 3': ['value chain emissions', 'scope three', 'scope-3', 'S3', 'other indirect emissions'],
    'tco2e': ['tonnes of CO2 equivalent', 'tons CO2e', 'metric tons CO2e', 'tCO2e'],
    
    # Energy terms
    'energy consumption': ['energy use', 'energy usage', 'total energy', 'electricity consumption', 'power consumption'],
    'renewable energy': ['clean energy', 'green energy', 'sustainable energy', 'renewable sources'],
    'mwh': ['megawatt hours', 'MWh', 'megawatt-hours'],
    
    # Water & waste terms
    'water withdrawal': ['water consumption', 'water usage', 'water use', 'freshwater withdrawal'],
    'waste recycled': ['recycling rate', 'waste diversion rate', 'waste recovery rate'],
    'hazardous waste': ['toxic waste', 'dangerous waste', 'special waste'],
    
    # Social terms
    'employee turnover': ['turnover rate', 'attrition rate', 'staff turnover', 'employee attrition'],
    'female representation': ['women in workforce', 'gender diversity', 'women employees', 'female employees'],
    'training hours': ['learning hours', 'development hours', 'skill training hours'],
    'lost time injury': ['ltifr', 'ltir', 'workplace injury rate', 'safety incident rate'],
    'employee satisfaction': ['engagement score', 'employee engagement', 'satisfaction rate'],
    'community investment': ['csr spend', 'social investment', 'csr expenditure', 'community spending'],
    
    # Governance terms
    'board independence': ['independent directors', 'independent board members', 'non-executive directors'],
    'female directors': ['women on board', 'women directors', 'board gender diversity'],
    'ceo pay ratio': ['compensation ratio', 'executive pay ratio', 'pay disparity ratio'],
    'ethics training': ['compliance training', 'code of conduct training', 'anti-corruption training'],
    'whistleblower': ['grievance', 'ethical complaints', 'integrity hotline'],
    
    # General ESG terms
    'report': ['disclosure', 'statement', 'filing'],
    'sustainability': ['esg', 'responsible business', 'corporate responsibility'],
    'fiscal year': ['financial year', 'fy', 'reporting year'],
    'total': ['aggregate', 'combined', 'overall', 'cumulative'],
    'reduced': ['decreased', 'lowered', 'cut', 'minimized'],
    'increased': ['grew', 'rose', 'expanded', 'improved'],
}

# ─── ESG SENTENCE TEMPLATES ─────────────────────────────────────────────────

METRIC_TEMPLATES = {
    'ghg_emissions': [
        "Total greenhouse gas emissions were {value} {unit} during the reporting period.",
        "The company reported GHG emissions of {value} {unit} for FY{year}.",
        "Our carbon footprint amounted to {value} {unit} in the current fiscal year.",
        "Aggregate GHG emissions stood at {value} {unit}, reflecting our operational impact.",
        "The organization's total carbon emissions reached {value} {unit}.",
    ],
    'scope1_emissions': [
        "Scope 1 emissions: {value} {unit} from direct operations.",
        "Direct emissions (Scope 1) totaled {value} {unit}.",
        "Our Scope 1 GHG emissions amounted to {value} {unit}.",
        "Direct operational emissions were recorded at {value} {unit} (Scope 1).",
        "S1 emissions from owned/controlled sources: {value} {unit}.",
    ],
    'scope2_emissions': [
        "Scope 2 emissions: {value} {unit} from purchased electricity.",
        "Indirect emissions (Scope 2) were {value} {unit}.",
        "Our Scope 2 emissions from electricity consumption totaled {value} {unit}.",
        "Purchased energy emissions (Scope 2): {value} {unit}.",
        "Indirect GHG emissions from energy were {value} {unit} (Scope 2).",
    ],
    'scope3_emissions': [
        "Scope 3 emissions: {value} {unit} across the value chain.",
        "Value chain emissions (Scope 3) amounted to {value} {unit}.",
        "Our Scope 3 other indirect emissions totaled {value} {unit}.",
        "Supply chain and downstream emissions (Scope 3): {value} {unit}.",
        "S3 emissions from business travel, commuting, and supply chain: {value} {unit}.",
    ],
    'co2_emissions': [
        "CO2 emissions: {value} {unit}.",
        "Carbon dioxide emissions were {value} {unit}.",
        "Total CO2e output reached {value} {unit} during the period.",
        "The company's carbon dioxide emissions were recorded at {value} {unit}.",
    ],
    'energy_consumption': [
        "Total energy consumption was {value} {unit}.",
        "Energy use: {value} {unit} across all operations.",
        "The company consumed {value} {unit} of energy in the reporting year.",
        "Energy usage across operations totaled {value} {unit}.",
        "Aggregate electricity and fuel consumption: {value} {unit}.",
    ],
    'renewable_energy': [
        "{value}% of energy sourced from renewable sources.",
        "Renewable energy share: {value}%.",
        "Clean energy accounted for {value}% of total consumption.",
        "The proportion of green energy in our mix was {value}%.",
        "{value}% of electricity came from solar, wind, and other renewable sources.",
    ],
    'water_withdrawal': [
        "Water withdrawal: {value} {unit}.",
        "Total water consumption was {value} {unit}.",
        "Freshwater usage: {value} {unit} during the reporting period.",
        "The company withdrew {value} {unit} of water from various sources.",
    ],
    'waste_recycled': [
        "{value}% of waste was recycled.",
        "Waste recycling rate: {value}%.",
        "The company achieved a {value}% waste diversion rate.",
        "Recycling and recovery rate stood at {value}%.",
    ],
    'hazardous_waste': [
        "Hazardous waste generated: {value} {unit}.",
        "The company produced {value} {unit} of hazardous waste.",
        "Toxic waste volume: {value} {unit}.",
        "Total hazardous waste: {value} {unit} during the reporting period.",
    ],
    'employee_turnover': [
        "Employee turnover rate: {value}%.",
        "Staff attrition rate was {value}%.",
        "The company's employee turnover stood at {value}%.",
        "Voluntary and involuntary turnover rate: {value}%.",
    ],
    'female_representation': [
        "Female representation in workforce: {value}%.",
        "Women comprised {value}% of the total workforce.",
        "Gender diversity: {value}% women employees.",
        "The share of female employees was {value}%.",
    ],
    'training_hours': [
        "Average training hours per employee: {value} hours.",
        "Employees received an average of {value} hours of training.",
        "Training and development: {value} hours per employee on average.",
        "Learning and development hours per employee: {value}.",
    ],
    'lost_time_injury': [
        "Lost time injury frequency rate: {value}.",
        "LTIFR: {value} per million man-hours.",
        "Workplace safety: LTIR of {value}.",
        "Lost time injury rate was recorded at {value}.",
    ],
    'employee_satisfaction': [
        "Employee satisfaction: {value}%.",
        "Employee engagement score: {value}%.",
        "Staff satisfaction survey result: {value}%.",
        "The engagement score reached {value}%.",
    ],
    'community_investment': [
        "Community investment: {value} crore.",
        "CSR expenditure: INR {value} crore.",
        "Social investment totaled {value} crore.",
        "The company invested {value} crore in community development programs.",
    ],
    'board_independence': [
        "Board independence: {value}%.",
        "{value}% of the board consists of independent directors.",
        "Independent board members: {value}%.",
        "Non-executive independent directors comprise {value}% of the board.",
    ],
    'female_directors': [
        "Female directors: {value}%.",
        "Women on board: {value}%.",
        "{value}% of board members are women.",
        "Board gender diversity: {value}% female directors.",
    ],
    'ceo_pay_ratio': [
        "CEO pay ratio: {value}:1.",
        "Executive compensation ratio: {value} to 1.",
        "The CEO-to-median-employee pay ratio was {value}:1.",
        "Pay disparity ratio: {value}:1.",
    ],
    'ethics_training': [
        "Ethics training completion: {value}%.",
        "Compliance training coverage: {value}%.",
        "{value}% of employees completed ethics and compliance training.",
        "Code of conduct training completion rate: {value}%.",
    ],
    'whistleblower_cases': [
        "Whistleblower cases reported: {value}.",
        "{value} complaints were received through the whistleblower mechanism.",
        "Grievance reports: {value} cases.",
        "Ethics hotline received {value} reports during the year.",
    ],
}


# ─── AUGMENTATION FUNCTIONS ─────────────────────────────────────────────────

def _synonym_replace(text: str, num_replacements: int = 2) -> str:
    """Replace random ESG terms with their synonyms."""
    result = text
    text_lower = text.lower()
    
    # Find which terms are present in the text
    replaceable = []
    for term, synonyms in ESG_SYNONYMS.items():
        if term in text_lower:
            replaceable.append((term, synonyms))
    
    if not replaceable:
        return result
    
    # Replace up to num_replacements terms
    chosen = random.sample(replaceable, min(num_replacements, len(replaceable)))
    for term, synonyms in chosen:
        synonym = random.choice(synonyms)
        # Case-preserving replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(synonym, result, count=1)
    
    return result


def _perturb_numbers(text: str, variation: float = 0.05) -> str:
    """Slightly vary numeric values (±5%) to teach model to generalize."""
    def _perturb_match(match):
        try:
            # Don't perturb years or very small numbers
            num_str = match.group(0).replace(',', '')
            num = float(num_str)
            if 1900 <= num <= 2100:  # looks like a year
                return match.group(0)
            if num == 0:
                return match.group(0)
            
            # Apply random perturbation
            factor = 1 + random.uniform(-variation, variation)
            new_num = num * factor
            
            # Preserve format (commas, decimals)
            if '.' in match.group(0):
                decimals = len(match.group(0).split('.')[-1])
                return f"{new_num:.{decimals}f}"
            elif ',' in match.group(0):
                return f"{int(new_num):,}"
            else:
                return str(int(new_num))
        except (ValueError, OverflowError):
            return match.group(0)
    
    return re.sub(r'[\d,]+\.?\d*', _perturb_match, text)


def _shuffle_sentences(text: str) -> str:
    """Reorder sentences within a chunk while keeping structure."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 2:
        return text
    
    # Keep first sentence in place (often has context), shuffle the rest
    rest = sentences[1:]
    random.shuffle(rest)
    return sentences[0] + ' ' + ' '.join(rest)


def _generate_from_template(metric_name: str, value, unit: str) -> Optional[str]:
    """Generate a synthetic sample from predefined templates."""
    templates = METRIC_TEMPLATES.get(metric_name)
    if not templates:
        return None
    
    template = random.choice(templates)
    
    # Format value
    if isinstance(value, float):
        if value.is_integer():
            formatted_value = f"{int(value):,}"
        else:
            formatted_value = f"{value:,.2f}"
    elif isinstance(value, int):
        formatted_value = f"{value:,}"
    else:
        formatted_value = str(value)
    
    year = random.choice(['2022-23', '2023-24', '2023', '2024', '2022'])
    
    return template.format(value=formatted_value, unit=unit, year=year)


def augment_single_sample(sample: Dict) -> List[Dict]:
    """
    Generate multiple augmented versions of a single sample.
    
    Returns 3-5 augmented samples per input.
    """
    augmented = []
    text = sample['text']
    metric_name = sample.get('metric_name', 'no_metric')
    
    if metric_name == 'no_metric':
        return []  # Don't augment negative samples
    
    # 1. Synonym replacement
    syn_text = _synonym_replace(text, num_replacements=2)
    if syn_text != text:
        aug_sample = deepcopy(sample)
        aug_sample['text'] = syn_text
        aug_sample['augmented'] = True
        augmented.append(aug_sample)
    
    # 2. Number perturbation
    perturbed_text = _perturb_numbers(text)
    if perturbed_text != text:
        aug_sample = deepcopy(sample)
        aug_sample['text'] = perturbed_text
        aug_sample['augmented'] = True
        augmented.append(aug_sample)
    
    # 3. Synonym + number perturbation combo
    combo_text = _perturb_numbers(_synonym_replace(text, num_replacements=1))
    if combo_text != text:
        aug_sample = deepcopy(sample)
        aug_sample['text'] = combo_text
        aug_sample['augmented'] = True
        augmented.append(aug_sample)
    
    # 4. Sentence shuffle
    shuffled_text = _shuffle_sentences(text)
    if shuffled_text != text:
        aug_sample = deepcopy(sample)
        aug_sample['text'] = shuffled_text
        aug_sample['augmented'] = True
        augmented.append(aug_sample)
    
    # 5. Template-based generation (if value available)
    value = sample.get('value')
    unit = sample.get('unit', '')
    if value is not None:
        template_text = _generate_from_template(metric_name, value, unit)
        if template_text:
            aug_sample = deepcopy(sample)
            aug_sample['text'] = template_text
            aug_sample['augmented'] = True
            augmented.append(aug_sample)
        noisy_text = _add_pdf_noise(text)
        if noisy_text != text:
            aug_sample = deepcopy(sample)
            aug_sample['text'] = noisy_text
            aug_sample['augmented'] = True
            augmented.append(aug_sample)
    return augmented


def augment_samples(samples: List[Dict], multiplier: int = 3) -> List[Dict]:
    """
    Augment a list of training samples.
    
    Args:
        samples: List of labeled sample dicts (positive only)
        multiplier: Target augmentation multiplier (3 = 3x more samples)
        
    Returns:
        List of augmented samples (does not include originals)
    """
    all_augmented = []
    
    for sample in samples:
        augs = augment_single_sample(sample)
        # Limit to multiplier per sample
        selected = random.sample(augs, min(multiplier, len(augs))) if len(augs) > multiplier else augs
        all_augmented.extend(selected)
    
    print(f"  [Augmentation] Generated {len(all_augmented)} augmented samples from {len(samples)} originals")
    return all_augmented


if __name__ == "__main__":
    # Quick test
    test_sample = {
        'text': 'Our total greenhouse gas emissions were 125,000 tCO2e in FY2023-24. '
                'Scope 1 emissions: 45,000 tCO2e. Renewable energy usage: 42%.',
        'metric_name': 'ghg_emissions',
        'category': 'environmental',
        'value': 125000,
        'unit': 'tCO2e',
        'sector': 'Tech',
    }
    
    print("Original:")
    print(f"  {test_sample['text'][:100]}...")
    
    augmented = augment_single_sample(test_sample)
    print(f"\nGenerated {len(augmented)} augmentations:")
    for i, aug in enumerate(augmented):
        print(f"  [{i+1}] {aug['text'][:100]}...")
