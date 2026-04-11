# ============================================================
# backend/ml/sarcasm_detector.py
#
# Sarcasm & Irony Detection
# ─────────────────────────
# Identifies sarcastic/ironic statements which may be:
# 1. True statements expressed sarcastically
# 2. False claims stated ironically
# 3. Humorous takes on misinformation
#
# Important for context-aware misinformation detection.
# ============================================================

import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# Sarcasm indicators
SARCASM_PATTERNS = [
    r"(yeah right|sure|oh wow|great|brilliant|fantastic)\W",  # Obvious sarcasm
    r"(because\s+)?that.?s exactly what we need",
    r"(nothing says|totally)*.*like",
    r"i'm sure .{5,50} (believes|accepts|agrees)",
    r"obviously.*(?:stupid|genius|brilliant)",
    r"as if\W",
    r"(yeah|okay).*\(.*\)",  # Text in parentheses often is ironic
]

# Irony indicators
IRONY_PATTERNS = [
    r"(not\.{0,3}|yet|but)\s+(?:actually|really)(,|\s)",
    r"(ironically|paradoxically|funnily enough|oddly enough)",
    r"they.{0,20}(say|claim|pretend).*truth.{0,20}",
    r"the.*is.* (exactly|precisely) opposite",
]

# Quotation marks often indicate sarcasm
QUOTATION_PATTERNS = r'"[^"]{10,200}"'  # "quoted sarcasm"

# ALL CAPS sections (especially short ones) can be ironic
CAPS_PATTERNS = r'\b[A-Z]{3,}\b'  # ALL CAPS words

# Contradictory statements (e.g., "X is Y" but "X is not Y")
CONTRADICTION_PATTERNS = [
    r"(?:is|are)\s+(not\s+)*(\w+).*(?:is|are)\s+(not\s+)*(?!\2)",
]


def detect_sarcasm(text: str) -> Dict[str, Any]:
    """
    Detect sarcasm and irony in text.
    
    Returns:
    {
        "is_sarcastic": bool,
        "sarcasm_score": float (0-1),
        "is_ironic": bool,
        "irony_score": float (0-1),
        "combined_score": float (0-1),
        "patterns_found": list,
        "details": str,
        "recommendation": str,
    }
    """
    text_lower = text.lower()
    sarcasm_hits = 0
    irony_hits = 0
    patterns_found = []
    
    # Check sarcasm patterns
    for pattern in SARCASM_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            sarcasm_hits += 2
            patterns_found.append(f"Sarcasm pattern: {pattern[:40]}...")
    
    # Check irony patterns
    for pattern in IRONY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            irony_hits += 2
            patterns_found.append(f"Irony pattern: {pattern[:40]}...")
    
    # Check for quotations (0.3 likelihood per quote)
    quotes = re.findall(QUOTATION_PATTERNS, text)
    sarcasm_hits += len(quotes) * 0.5
    if quotes:
        patterns_found.append(f"Quotation marks ({len(quotes)} found)")
    
    # Check for odd capitalization
    caps_words = re.findall(CAPS_PATTERNS, text)
    if len(caps_words) / max(1, len(text.split())) > 0.15:  # >15% all caps
        sarcasm_hits += 1.5
        patterns_found.append(f"Excessive capitalization ({len(caps_words)} words)")
    
    # Check for contradictions
    for pattern in CONTRADICTION_PATTERNS:
        if len(re.findall(pattern, text_lower)) > 0:
            irony_hits += 1
            patterns_found.append("Contradiction detected")
    
    # Normalize scores
    sarcasm_score = min(1.0, sarcasm_hits / 5.0)
    irony_score = min(1.0, irony_hits / 4.0)
    combined_score = max(sarcasm_score, irony_score)
    
    is_sarcastic = sarcasm_score > 0.4
    is_ironic = irony_score > 0.3
    
    # Generate details
    details = ""
    if is_sarcastic and is_ironic:
        details = "This statement appears to contain both sarcasm and irony. " \
                  "The intended meaning may be the opposite of the literal text. " \
                  "Context analysis is needed to verify the actual claim."
    elif is_sarcastic:
        details = "This statement appears sarcastic. " \
                  "The actual meaning may differ from the literal text. " \
                  "Consider the broader context and tone."
    elif is_ironic:
        details = "This statement appears ironic with contradictory elements. " \
                  "The implied meaning may be different from what's explicitly stated."
    else:
        details = "No significant sarcasm or irony detected. Text appears straightforward."
    
    recommendation = ""
    if combined_score > 0.6:
        recommendation = "⚠️ HIGH: Requires manual review for actual intent. " \
                        "Automated misinformation scoring may be unreliable for sarcastic text."
    elif combined_score > 0.3:
        recommendation = "⚠️ MEDIUM: May be sarcastic. Cross-reference with other signals."
    else:
        recommendation = "✅ LOW: Appears genuine. Standard misinformation checks apply."
    
    return {
        "is_sarcastic": is_sarcastic,
        "sarcasm_score": round(sarcasm_score, 3),
        "is_ironic": is_ironic,
        "irony_score": round(irony_score, 3),
        "combined_score": round(combined_score, 3),
        "patterns_found": patterns_found,
        "details": details,
        "recommendation": recommendation,
    }


def estimate_literal_meaning(text: str) -> Dict[str, Any]:
    """
    If text is sarcastic, try to estimate the literal (intended) meaning.
    
    Returns:
    {
        "likely_literal": str,  # Inversed/corrected statement
        "confidence": float,
    }
    """
    sarcasm_info = detect_sarcasm(text)
    
    if sarcasm_info["combined_score"] < 0.4:
        return {
            "likely_literal": text,  # No sarcasm detected
            "confidence": 0.0,
        }
    
    # For high-confidence sarcasm, try to invert
    text_lower = text.lower()
    
    # Common sarcasm inversions
    inversions = {
        r"\byeah(?:\s+|,)": "No, ",
        r"\bsure(?:\s+|,)": "Actually, ",
        r"\bgreat\b": "terrible",
        r"\bfantastic\b": "awful",
        r"\blovely\b": "horrible",
        r"\bwonderful\b": "terrible",
        r"\bbrilliant\b": "stupid",
        r"\bamazing\b": "disappointing",
    }
    
    likely_literal = text
    for sarcastic, inverted in inversions.items():
        likely_literal = re.sub(sarcastic, inverted, likely_literal, flags=re.IGNORECASE)
    
    confidence = min(0.7, sarcasm_info["combined_score"])
    
    return {
        "likely_literal": likely_literal,
        "confidence": round(confidence, 3),
    }
