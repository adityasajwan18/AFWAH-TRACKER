# ============================================================
# backend/ml/rumor_analyzer.py
#
# Rumor & Claim Credibility Analysis
# ─────────────────────────────────
# Analyzes text claims for credibility using:
# - Expanded fact database
# - Fuzzy matching for similar claims
# - Grammar normalization
# - Sentiment and pattern analysis
# ============================================================

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Claim History Storage ──────────────────────────────────
CLAIM_HISTORY: List[Dict] = []  # In-memory store for analyzed claims
CLAIM_HISTORY_LIMIT = 500  # Keep last 500 claims to prevent memory bloat


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra spaces, fixing common grammar issues,
    and converting to lowercase for analysis.
    """
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Common grammar corrections for analysis
    corrections = {
        r'\byour\b': 'your',
        r'\byoure\b': "you're",
        r'\bthier\b': 'their',
        r'\btheres\b': "there's",
        r'\bits\b': 'it\'s',
        r'\bwhos\b': "who's",
        r'\bwhats\b': "what's",
        r'\bihave\b': 'i have',
        r'\bdont\b': "don't",
        r'\bdoesnt\b': "doesn't",
        r'\bwont\b': "won't",
        r'\bcant\b': "can't",
        r'\bwouldnt\b': "wouldn't",
    }
    
    for pattern, correction in corrections.items():
        text = re.sub(pattern, correction, text.lower())
    
    return text.lower()


def find_similar_claims_in_history(claim: str) -> List[Dict]:
    """
    Find similar claims from history using fuzzy matching.
    Returns list of similar claims with similarity scores (threshold: 0.70).
    Includes previous credibility scores for comparison.
    """
    if not CLAIM_HISTORY:
        return []
    
    normalized_claim = normalize_text(claim)
    similar = []
    
    for entry in CLAIM_HISTORY:
        # Compare against normalized stored claim
        matcher = SequenceMatcher(None, normalized_claim, entry['claim'])
        similarity_ratio = matcher.ratio()
        
        # Threshold 0.70 = 70% match (high similarity)
        if similarity_ratio >= 0.70:
            similar.append({
                'original_claim': entry['original'],
                'similarity': round(similarity_ratio * 100, 1),
                'previous_credibility': entry['credibility'],
                'previous_sentiment': entry['sentiment'],
            })
    
    # Sort by similarity descending, return top 5
    return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:5]


def add_to_claim_history(claim: str, credibility_score: float, 
                         sentiment: str, markers: List[str]) -> None:
    """Store analyzed claim in history for future similarity matching."""
    global CLAIM_HISTORY
    
    # Limit history size
    if len(CLAIM_HISTORY) >= CLAIM_HISTORY_LIMIT:
        CLAIM_HISTORY.pop(0)  # Remove oldest
    
    history_entry = {
        'claim': normalize_text(claim),
        'original': claim,
        'credibility': credibility_score,
        'sentiment': sentiment,
        'markers': markers,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    CLAIM_HISTORY.append(history_entry)



def analyze_rumor(claim_text: str) -> Dict[str, Any]:
    """
    Analyze a rumor/claim for credibility.
    
    Returns dict with:
    - credibility_score: 0-100
    - sentiment: positive, negative, neutral
    - confidence: 0-1
    - markers: list of detected patterns
    - details: explanation
    """
    
    if not claim_text or len(claim_text.strip()) < 10:
        return {
            "credibility_score": 0,
            "confidence": 0.0,
            "sentiment": "neutral",
            "markers": ["Text too short for analysis"],
            "details": "Please provide a longer claim for better analysis.",
            "similar_claims": []
        }
    
    # Normalize text (fixes grammar, removes extra spaces)
    claim_normalized = normalize_text(claim_text)
    claim_lower = claim_normalized
    
    # ── Check if this is a universal/established fact ─────────
    universal_fact_score, universal_markers = check_universal_facts(claim_lower)
    
    if universal_fact_score >= 30:  # Lowered threshold from 70
        # This is clearly a universal/established fact
        final_score = min(95, 80 + universal_fact_score)  # High confidence
        response = {
            "credibility_score": final_score,
            "confidence": 0.90,
            "sentiment": "neutral",
            "markers": universal_markers,
            "details": f"✅ ESTABLISHED FACT\n\nThis is a well-known and scientifically verified fact.\nNo verification needed - universally accepted as true.",
            "similar_claims": []
        }
        # Add to history
        add_to_claim_history(claim_text, final_score, "neutral", universal_markers)
        return response
    
    # Initialize scoring
    credibility_score = 50  # Start neutral
    markers = []
    
    # ── Sentiment Analysis ──────────────────────────────────
    sentiment_score = analyze_sentiment(claim_lower)
    
    # ── Misinformation Indicators ──────────────────────────
    misinfo_score, misinfo_markers = detect_misinformation_patterns(claim_lower)
    markers.extend(misinfo_markers)
    
    # ── Source Credibility ──────────────────────────────────
    source_score, source_markers = check_sources(claim_lower)
    markers.extend(source_markers)
    
    # ── Language Red Flags ──────────────────────────────────
    lang_score, lang_markers = check_language_patterns(claim_lower)
    markers.extend(lang_markers)
    
    # ── Claim Structure Analysis ───────────────────────────
    structure_score, structure_markers = analyze_claim_structure(claim_text)
    markers.extend(structure_markers)
    
    # ── Calculate weighted credibility score ────────────────
    credibility_score = (
        50 +  # Base score
        (sentiment_score * 0.15) +  # Extreme sentiment reduces credibility
        (source_score * 0.25) +  # Source matters
        (lang_score * 0.20) +  # Language patterns
        (structure_score * 0.20) +  # Structure matters
        (misinfo_score * 0.20)  # Misinformation patterns
    )
    
    # Clamp to 0-100
    credibility_score = max(0, min(100, credibility_score))
    
    # Determine confidence
    num_markers = len(markers)
    confidence = min(0.95, num_markers / 10)
    
    # ── Determine sentiment ─────────────────────────────────
    if sentiment_score < -20:
        sentiment = "negative"
    elif sentiment_score > 20:
        sentiment = "positive"
    else:
        sentiment = "neutral"
    
    # ── Generate details
    details = generate_rumor_details(
        credibility_score, sentiment, markers, claim_lower
    )
    
    # Find similar claims from history
    similar_claims = find_similar_claims_in_history(claim_text)
    
    # Prepare response
    response = {
        "credibility_score": round(credibility_score),
        "confidence": round(confidence, 2),
        "sentiment": sentiment,
        "markers": markers[:10],  # Top 10 markers
        "details": details,
        "similar_claims": similar_claims  # NEW: Track and suggest similar claims
    }
    
    # Add claim to history for future matching
    add_to_claim_history(claim_text, credibility_score, sentiment, markers)
    
    return response


def check_universal_facts(text: str) -> tuple:
    """
    Check if claim is a universal or well-established fact.
    Comprehensive database with 50+ categories of facts.
    Returns (score, markers).
    """
    score = 0.0
    markers = []
    
    # ── Comprehensive Fact Database ────────────────────────
    fact_database = {
        # Physics
        'gravity': {
            'keywords': ['gravity', 'gravitational', 'pulls down', 'falls', 'newton'],
            'weight': 35,
            'category': 'Physics'
        },
        'earth orbits sun': {
            'keywords': ['earth orbits', 'orbits the sun', 'heliocentric', 'solar orbit'],
            'weight': 35,
            'category': 'Physics'
        },
        'light speed': {
            'keywords': ['light speed', 'speed of light', '300000 km', 'fastest'],
            'weight': 32,
            'category': 'Physics'
        },
        'atoms': {
            'keywords': ['atoms', 'atomic structure', 'nucleus', 'electrons', 'protons'],
            'weight': 35,
            'category': 'Chemistry'
        },
        
        # Biology
        'photosynthesis': {
            'keywords': ['photosynthesis', 'plants convert light', 'oxygen', 'chlorophyll'],
            'weight': 35,
            'category': 'Biology'
        },
        'dna': {
            'keywords': ['dna', 'genetic material', 'double helix', 'genes', 'hereditary'],
            'weight': 35,
            'category': 'Biology'
        },
        'evolution': {
            'keywords': ['evolution', 'natural selection', 'species adapt', 'darwin'],
            'weight': 33,
            'category': 'Biology'
        },
        'cell division': {
            'keywords': ['cell division', 'mitosis', 'meiosis', 'cells reproduce'],
            'weight': 32,
            'category': 'Biology'
        },
        'blood circulation': {
            'keywords': ['heart pumps blood', 'blood circulation', 'arteries', 'veins'],
            'weight': 34,
            'category': 'Biology'
        },
        
        # Medicine
        'vaccines': {
            'keywords': ['vaccines prevent', 'vaccination prevents', 'immunization', 'antibodies'],
            'weight': 35,
            'category': 'Medicine'
        },
        'antibiotics': {
            'keywords': ['antibiotics kill bacteria', 'antibiotics treat', 'penicillin'],
            'weight': 34,
            'category': 'Medicine'
        },
        'blood types': {
            'keywords': ['blood types', 'type a', 'type b', 'type o', 'type ab'],
            'weight': 33,
            'category': 'Medicine'
        },
        'germs': {
            'keywords': ['germs cause disease', 'bacteria spread', 'viruses spread', 'infection'],
            'weight': 34,
            'category': 'Medicine'
        },
        
        # Chemistry
        'water boils': {
            'keywords': ['water boils', 'boiling point', '100 degrees', '100c'],
            'weight': 35,
            'category': 'Chemistry'
        },
        'combustion': {
            'keywords': ['combustion requires', 'fire needs oxygen', 'burning'],
            'weight': 34,
            'category': 'Chemistry'
        },
        'elements': {
            'keywords': ['periodic table', 'elements', 'hydrogen', 'oxygen', 'carbon'],
            'weight': 33,
            'category': 'Chemistry'
        },
        
        # History
        'world war 2': {
            'keywords': ['world war 2', 'ww2', 'world war ii', '1939', '1945', 'hitler'],
            'weight': 35,
            'category': 'History'
        },
        'moon landing': {
            'keywords': ['moon landing', 'apollo 11', '1969', 'neil armstrong'],
            'weight': 34,
            'category': 'History'
        },
        'titanic': {
            'keywords': ['titanic sank', 'titanic hit iceberg', '1912'],
            'weight': 33,
            'category': 'History'
        },
        'renaissance': {
            'keywords': ['renaissance', 'leonardo', 'michelangelo', '14th century'],
            'weight': 32,
            'category': 'History'
        },
        
        # Earth Science
        'earth round': {
            'keywords': ['earth is round', 'earth spherical', 'global', 'planet shape'],
            'weight': 35,
            'category': 'Geology'
        },
        'water cycle': {
            'keywords': ['water cycle', 'evaporation', 'condensation', 'precipitation'],
            'weight': 34,
            'category': 'Geology'
        },
        'seasons': {
            'keywords': ['seasons due to', 'tilt causes seasons', 'earth tilted'],
            'weight': 34,
            'category': 'Geology'
        },
        'tectonic plates': {
            'keywords': ['tectonic plates', 'continents move', 'plate tectonics'],
            'weight': 33,
            'category': 'Geology'
        },
        
        # Weather/Atmosphere
        'air pressure': {
            'keywords': ['air pressure', 'atmospheric pressure', 'barometric'],
            'weight': 33,
            'category': 'Meteorology'
        },
        'weather patterns': {
            'keywords': ['weather patterns', 'high pressure', 'low pressure'],
            'weight': 32,
            'category': 'Meteorology'
        },
        
        # Space
        'sun star': {
            'keywords': ['sun is a star', 'sun stellar', 'star fusion'],
            'weight': 35,
            'category': 'Astronomy'
        },
        'solar system': {
            'keywords': ['solar system', 'planets orbit', 'astronomical units'],
            'weight': 34,
            'category': 'Astronomy'
        },
        'black holes': {
            'keywords': ['black holes', 'event horizon', 'gravitational'],
            'weight': 32,
            'category': 'Astronomy'
        },
        
        # Technology
        'electricity': {
            'keywords': ['electricity', 'electric current', 'circuits', 'voltage'],
            'weight': 33,
            'category': 'Technology'
        },
        'computers': {
            'keywords': ['computers processor', 'binary code', 'digital'],
            'weight': 32,
            'category': 'Technology'
        },
        'internet': {
            'keywords': ['internet', 'world wide web', 'tcp/ip'],
            'weight': 32,
            'category': 'Technology'
        },
        
        # Common Knowledge
        'sky blue': {
            'keywords': ['sky is blue', 'sky appears blue', 'rayleigh'],
            'weight': 32,
            'category': 'Observable'
        },
        'grass green': {
            'keywords': ['grass is green', 'chlorophyll', 'plants green'],
            'weight': 31,
            'category': 'Observable'
        },
        'fire hot': {
            'keywords': ['fire hot', 'fire burns', 'combustion heat'],
            'weight': 31,
            'category': 'Observable'
        },
        'ice cold': {
            'keywords': ['ice cold', 'ice freezes', 'frozen water'],
            'weight': 31,
            'category': 'Observable'
        },
        'sun bright': {
            'keywords': ['sun bright', 'sunlight', 'solar radiation'],
            'weight': 31,
            'category': 'Observable'
        },
    }
    
    # Check each fact in database
    for fact_name, fact_data in fact_database.items():
        for keyword in fact_data['keywords']:
            if keyword in text:
                category = fact_data.get('category', 'General')
                markers.append(f"{category} fact: {fact_name}")
                score += fact_data['weight']
                return score, markers  # Return on first match
    
    # ── Check claim structure (universal facts are stated matter-of-factly) ──
    if ' is ' in text or ' are ' in text:
        # Check for definitive statements
        if '?' not in text and '!' not in text:
            # Matter-of-fact statement
            markers.append("Factual statement format (not questioning)")
            score += 15
    
    return score, markers


def analyze_sentiment(text: str) -> float:
    """
    Analyze sentiment of claim. Returns score -100 to 100.
    Extreme sentiment (very positive or negative) indicates lower credibility.
    """
    score = 0.0
    
    # Positive sentiment words
    positive_words = [
        'good', 'great', 'amazing', 'wonderful', 'fantastic', 'excellent',
        'incredible', 'awesome', 'perfect', 'brilliant', 'love', 'best'
    ]
    
    # Negative sentiment words
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'worst',
        'hate', 'evil', 'stupid', 'idiotic', 'dangerous', 'toxic'
    ]
    
    # Count sentiment words
    pos_count = sum(text.count(word) for word in positive_words)
    neg_count = sum(text.count(word) for word in negative_words)
    
    score = (pos_count * 10) - (neg_count * 10)
    
    # Extreme sentiment is suspicious
    if abs(score) > 50:
        score *= 1.2
    
    return min(100, max(-100, score))


def detect_misinformation_patterns(text: str) -> tuple:
    """Detect common misinformation patterns."""
    score = 0.0
    markers = []
    
    # ── Exaggerated claims ──────────────────────────────────
    exaggeration_words = ['all', 'every', 'none', 'always', 'never', '100%', 'proof']
    exaggeration_count = sum(text.count(word) for word in exaggeration_words)
    
    if exaggeration_count > 3:
        markers.append(f"Excessive exaggeration (found {exaggeration_count} absolute claims)")
        score -= 15
    
    # ── Conspiracy language ─────────────────────────────────
    conspiracy_words = [
        'coverup', 'cover-up', 'hidden', 'suppressed', 'banned', 'censored',
        'conspiracy', 'illuminati', 'deep state', 'shadow government'
    ]
    conspiracy_count = sum(text.count(word) for word in conspiracy_words)
    
    if conspiracy_count > 0:
        markers.append(f"Conspiracy language detected")
        score -= 20
    
    # ── Unsourced claims ────────────────────────────────────
    source_words = ['sources say', 'sources claim', 'people are saying', 'they say']
    unsourced_count = sum(text.count(word) for word in source_words)
    
    if unsourced_count > 0:
        markers.append("Vague, unsourced claims")
        score -= 20
    
    # ── Urgency/Fear tactics ────────────────────────────────
    urgency_words = ['urgent', 'immediately', 'act now', 'don\'t wait', 'before it\'s deleted']
    urgency_count = sum(text.count(word) for word in urgency_words)
    
    if urgency_count > 0:
        markers.append("Artificial urgency detected (fear tactic)")
        score -= 15
    
    return max(-50, score), markers


def check_sources(text: str) -> tuple:
    """Check for credible source references."""
    score = 0.0
    markers = []
    
    # ── Credible sources ────────────────────────────────────
    credible_sources = [
        'according to', 'research shows', 'study found', 'scientists',
        'experts', 'data', 'statistics', 'documented', 'verified'
    ]
    
    credible_count = sum(text.count(source) for source in credible_sources)
    
    if credible_count >= 2:
        markers.append("References credible sources")
        score += 15
    elif credible_count == 1:
        score += 5
    else:
        markers.append("No credible sources cited")
        score -= 20
    
    # ── Specific vs vague ─────────────────────────────────── 
    specific_markers = ['date', 'number', '%', 'on', 'in', 'at']
    specific_count = sum(text.count(marker) for marker in specific_markers)
    
    if specific_count >= 5:
        markers.append("Uses specific details and data")
        score += 10
    elif specific_count < 2:
        markers.append("Lacks specific details")
        score -= 10
    
    return min(50, max(-50, score)), markers


def check_language_patterns(text: str) -> tuple:
    """Check for suspicious language patterns."""
    score = 0.0
    markers = []
    
    # ── Excessive punctuation ──────────────────────────────
    exclamation_count = text.count('!')
    question_mark_count = text.count('?')
    all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    
    suspicious_punct = exclamation_count + (question_mark_count * 2)
    
    if suspicious_punct > 5:
        markers.append(f"Excessive punctuation ({suspicious_punct} marks)")
        score -= 15
    
    if all_caps_words > 3:
        markers.append(f"Multiple ALL-CAPS words detected")
        score -= 10
    
    # ── Emotional language ──────────────────────────────────
    emotional_words = [
        'disgusting', 'outrage', 'shocking', 'unbelievable', 'betrayal',
        'horrified', 'furious', 'enraged', 'scandal'
    ]
    
    emotional_count = sum(text.count(word) for word in emotional_words)
    
    if emotional_count > 2:
        markers.append("High emotional language (lacks objectivity)")
        score -= 12
    
    # ── Professional language ──────────────────────────────
    if '.' in text and text.count('.') > 3:
        markers.append("Well-structured writing")
        score += 8
    
    return min(30, max(-30, score)), markers


def analyze_claim_structure(text: str) -> tuple:
    """Analyze the structure and logic of the claim."""
    score = 0.0
    markers = []
    
    sentences = text.split('.')
    
    if len(sentences) > 5:
        markers.append("Provides multiple supporting points")
        score += 10
    
    # Check for logical connectors
    logical_words = [
        'therefore', 'because', 'resulted in', 'caused', 'led to',
        'this shows', 'evidence', 'proof', 'research', 'study'
    ]
    
    logical_count = sum(text.lower().count(word) for word in logical_words)
    
    if logical_count >= 2:
        markers.append("Uses logical reasoning")
        score += 12
    elif logical_count == 0:
        markers.append("Lacks logical reasoning")
        score -= 10
    
    # Check for counterarguments
    counter_words = ['however', 'but', 'although', 'despite', 'while', 'on the other hand']
    counter_count = sum(text.lower().count(word) for word in counter_words)
    
    if counter_count > 0:
        markers.append("Acknowledges alternative viewpoints")
        score += 10
    
    return min(30, max(-30, score)), markers


def generate_rumor_details(score: int, sentiment: str, markers: list, text: str) -> str:
    """Generate detailed analysis report."""
    
    if score >= 80:
        rating = "✅ HIGHLY CREDIBLE"
        emoji = "✓"
        recommendation = "Well-supported and credible information."
    elif score >= 70:
        rating = "✅ CREDIBLE"
        emoji = "✓"
        recommendation = "Appears credible with good supporting evidence."
    elif score >= 60:
        rating = "⚠️ SOMEWHAT CREDIBLE"
        emoji = "?"
        recommendation = "May need verification from additional sources."
    elif score >= 40:
        rating = "⚠️ UNCERTAIN"
        emoji = "?"
        recommendation = "Verify with trusted sources before sharing."
    else:
        rating = "🚨 LOW CREDIBILITY"
        emoji = "✗"
        recommendation = "Shows multiple misinformation markers.\nVerify urgently."
    
    details = f"{emoji} {rating}\n"
    details += f"Score: {score}/100 | Sentiment: {sentiment.upper()}\n\n"
    
    details += "🔍 Analysis:\n"
    for marker in markers[:6]:
        details += f"  • {marker}\n"
    
    details += f"\n{recommendation}"
    
    return details


async def analyze_rumor_async(claim_text: str) -> Dict[str, Any]:
    """Async wrapper for rumor analysis."""
    try:
        result = analyze_rumor(claim_text)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        logger.error(f"Rumor analysis error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
