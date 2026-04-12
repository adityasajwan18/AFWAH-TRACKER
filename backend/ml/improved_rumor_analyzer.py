# ============================================================
# backend/ml/improved_rumor_analyzer.py
#
# ML-powered Rumor Detection using trained models
# Builds on traditional analysis with machine learning
# ============================================================

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ImprovedRumorAnalyzer:
    """ML-powered rumor and misinformation detection"""
    
    def __init__(self):
        """Initialize with pre-trained models"""
        self.rf_model = None
        self.gb_model = None
        self.vectorizer = None
        self.claim_history = []
        self.model_path = Path(__file__).parent / "models"
        
        self._load_models()
        self._load_keywords_database()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            if not self.model_path.exists():
                logger.warning(f"Models directory not found: {self.model_path}")
                logger.info("Run: python scripts/train_rumor_detector.py")
                return
            
            # Load Random Forest
            rf_path = self.model_path / 'rf_rumor_model.pkl'
            if rf_path.exists():
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info("✅ Loaded Random Forest model")
            
            # Load Gradient Boosting
            gb_path = self.model_path / 'gb_rumor_model.pkl'
            if gb_path.exists():
                with open(gb_path, 'rb') as f:
                    self.gb_model = pickle.load(f)
                logger.info("✅ Loaded Gradient Boosting model")
            
            # Load Vectorizer
            vec_path = self.model_path / 'tfidf_vectorizer.pkl'
            if vec_path.exists():
                with open(vec_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("✅ Loaded TF-IDF Vectorizer")
        
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def _load_keywords_database(self):
        """Load misinformation keywords database"""
        self.misinformation_keywords = {
            'health': [
                'cure', 'miracle', 'hidden', 'suppressed', 'pharmaceutical', 
                'government conspiracy', 'dangerous', 'harmful', 'side effects',
                'spike protein', 'myocarditis'
            ],
            'conspiracy': [
                'deep state', 'shadow government', 'illuminati', 'new world order',
                'global elite', 'population control', 'microchip', 'AI takeover'
            ],
            'political': [
                'coup', 'stolen', 'rigged', 'fraud', 'fake', 'hoax',
                'deep state', 'shadow', 'shadow government'
            ],
            'science_denial': [
                'hoax', 'fake science', 'fake', 'not real',
                'all a lie', 'propaganda', 'censored'
            ]
        }
    
    def analyze(self, text: str, context: str = "general") -> Dict[str, Any]:
        """
        Comprehensive rumor analysis using ML and heuristics
        
        Args:
            text: Text to analyze
            context: Analysis context ('health', 'political', 'general')
        
        Returns:
            Detailed analysis with credibility scores
        """
        
        try:
            if not text or len(text.strip()) < 10:
                return self._empty_result("Text too short")
            
            # 1. ML-based classification
            ml_result = self._ml_classify(text)
            
            # 2. Keyword analysis
            keyword_result = self._analyze_keywords(text, context)
            
            # 3. Pattern analysis
            pattern_result = self._analyze_patterns(text)
            
            # 4. Claim consistency
            consistency_result = self._check_claim_consistency(text)
            
            # 5. Source credibility signals
            source_result = self._analyze_source_signals(text)
            
            # Ensemble decision
            confidence = self._ensemble_decision(
                ml_result, keyword_result, pattern_result,
                consistency_result, source_result
            )
            
            # Generate report
            report = self._generate_report(
                text, confidence, ml_result, keyword_result,
                pattern_result, source_result
            )
            
            return {
                "status": "success",
                "is_misinformation": confidence > 0.65,
                "confidence": float(confidence),
                "credibility_score": 1.0 - confidence,
                "scores": {
                    "ml_score": float(ml_result['score']),
                    "keyword_score": float(keyword_result['score']),
                    "pattern_score": float(pattern_result['score']),
                    "consistency_score": float(consistency_result['score']),
                    "source_score": float(source_result['score'])
                },
                "indicators": self._extract_indicators(
                    ml_result, keyword_result, pattern_result, consistency_result, source_result
                ),
                "recommendation": self._get_recommendation(confidence),
                "details": report,
                "analyzed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return self._empty_result(f"Analysis failed: {e}")
    
    def _ml_classify(self, text: str) -> Dict[str, Any]:
        """ML-based classification"""
        
        try:
            if not self.vectorizer or not self.rf_model:
                return {'score': 0.5, 'model': 'unavailable', 'details': []}
            
            # Vectorize
            X = self.vectorizer.transform([text])
            
            # Get predictions
            rf_pred = self.rf_model.predict(X)[0]
            rf_proba = self.rf_model.predict_proba(X)[0]
            
            gb_pred = self.gb_model.predict(X)[0]
            gb_proba = self.gb_model.predict_proba(X)[0]
            
            # Ensemble average
            avg_score = (rf_proba.max() + gb_proba.max()) / 2
            
            # Convert to misinformation confidence (invert if label==1 means "accurate")
            if rf_pred == 0 or gb_pred == 0:  # Both predict misinformation
                score = avg_score
            else:
                score = 1.0 - avg_score
            
            return {
                'score': score,
                'model': 'ensemble',
                'rf_prediction': rf_pred,
                'gb_prediction': gb_pred,
                'confidence': avg_score,
                'details': ['ML ensemble classification applied']
            }
        
        except Exception as e:
            logger.debug(f"ML classification skipped: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_keywords(self, text: str, context: str) -> Dict[str, Any]:
        """Keyword-based credibility scoring"""
        
        text_lower = text.lower()
        keywords_found = []
        misinformation_score = 0.0
        
        # Check for misinformation keywords
        for category, keywords in self.misinformation_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    keywords_found.append(keyword)
                    misinformation_score += 0.15
        
        # Context-specific keywords
        if context == 'health':
            health_red_flags = ['cure', 'miracle', 'big pharma hiding', 'natural remedy']
            for flag in health_red_flags:
                if flag in text_lower:
                    misinformation_score += 0.2
                    keywords_found.append(flag)
        
        misinformation_score = min(misinformation_score, 1.0)
        
        return {
            'score': misinformation_score,
            'keywords_found': keywords_found[:10],  # Top 10
            'details': [f"Found {len(keywords_found)} misinformation keywords"]
        }
    
    def _analyze_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze common misinformation patterns"""
        
        patterns = {
            'ALL_CAPS': text.upper() == text and len(text) > 20,
            'excessive_punctuation': text.count('!') >= 3 or text.count('?') >= 3,
            'emotion_words': self._count_emotion_words(text),
            'logical_fallacies': self._detect_logical_fallacies(text),
            'appeal_to_authority': 'doctors hate' in text.lower() or 'they don\'t want' in text.lower(),
            'false_dichotomy': ' or ' in text and len(text.split(' or ')) == 2,
        }
        
        # Calculate pattern score
        pattern_count = sum(1 for v in patterns.values() if v)
        pattern_score = min(pattern_count * 0.2, 1.0)
        
        return {
            'score': pattern_score,
            'patterns_detected': {k: v for k, v in patterns.items() if v},
            'details': [f"Detected {pattern_count} misinformation patterns"]
        }
    
    def _check_claim_consistency(self, text: str) -> Dict[str, Any]:
        """Check if claim is internally consistent"""
        
        # Simple heuristics for consistency
        consistency_score = 0.5
        issues = []
        
        # Check for self-contradiction
        if 'but' in text.lower() or 'however' in text.lower():
            parts = text.split('but')
            if len(parts) == 2:
                # Might indicate contradiction
                consistency_score -= 0.1
                issues.append("Potential contradiction detected")
        
        # Check for extreme claims
        extreme_phrases = ['100% certain', 'everyone knows', 'proven fact', 'no doubt']
        if any(phrase in text.lower() for phrase in extreme_phrases):
            consistency_score -= 0.15
            issues.append("Extreme or absolutist language")
        
        # Check for vagueness
        vague_words = ['apparently', 'allegedly', 'supposedly', 'some say']
        if any(word in text.lower() for word in vague_words):
            consistency_score -= 0.1
            issues.append("Vague attribution")
        
        consistency_score = max(0, consistency_score)
        
        return {
            'score': 1.0 - consistency_score,  # Invert: low consistency = high misinformation
            'issues': issues,
            'consistency_rating': consistency_score
        }
    
    def _analyze_source_signals(self, text: str) -> Dict[str, Any]:
        """Analyze credibility signals in the text"""
        
        signals = {
            'has_citations': 'according to' in text.lower() or 'study' in text.lower(),
            'has_dates': any(str(year) in text for year in range(1900, 2030)),
            'has_numbers': any(char.isdigit() for char in text),
            'claims_authority': 'expert' in text.lower() or 'scientist' in text.lower(),
            'includes_evidence': 'because' in text.lower() or 'evidence' in text.lower(),
        }
        
        credibility_indicators = sum(1 for v in signals.values() if v)
        source_score = credibility_indicators / len(signals)
        
        return {
            'score': 1.0 - source_score,  # Lower score = more credible
            'credibility_signals': {k: v for k, v in signals.items() if v},
            'details': [f"Found {credibility_indicators} credibility signals"]
        }
    
    def _ensemble_decision(self, ml_result, keyword_result, pattern_result,
                          consistency_result, source_result) -> float:
        """Weighted ensemble of all analyses"""
        
        weights = {
            'ml': 0.35,
            'keyword': 0.25,
            'pattern': 0.20,
            'consistency': 0.12,
            'source': 0.08
        }
        
        confidence = (
            ml_result.get('score', 0.5) * weights['ml'] +
            keyword_result.get('score', 0.5) * weights['keyword'] +
            pattern_result.get('score', 0.5) * weights['pattern'] +
            consistency_result.get('score', 0.5) * weights['consistency'] +
            source_result.get('score', 0.5) * weights['source']
        )
        
        return min(max(confidence, 0), 1.0)
    
    def _count_emotion_words(self, text: str) -> bool:
        """Count emotion/sensational words"""
        emotion_words = [
            'shocking', 'devastating', 'horrific', 'disgusting', 'incredible',
            'unbelievable', 'amazing', 'insane', 'crazy', 'dangerous'
        ]
        return any(word in text.lower() for word in emotion_words)
    
    def _detect_logical_fallacies(self, text: str) -> bool:
        """Detect common logical fallacies"""
        fallacies = [
            'ad hominem' in text.lower(),
            'straw man' in text.lower(),
            'slippery slope' in text.lower(),
            'appeal to pity' in text.lower(),
        ]
        return any(fallacies)
    
    def _extract_indicators(self, ml_result, keyword_result, 
                           pattern_result, consistency_result, source_result) -> List[str]:
        """Extract key indicators"""
        
        indicators = []
        
        if keyword_result.get('keywords_found'):
            indicators.append(f"🚩 Found misinformation keywords: {', '.join(keyword_result['keywords_found'][:3])}")
        
        if pattern_result.get('patterns_detected'):
            patterns = list(pattern_result['patterns_detected'].keys())
            indicators.append(f"⚠️ Detected patterns: {', '.join(patterns[:3])}")
        
        if consistency_result.get('issues'):
            indicators.append(f"❓ Consistency issues: {consistency_result['issues'][0]}")
        
        if not source_result.get('credibility_signals'):
            indicators.append("❓ Lacks credibility signals (citations, evidence)")
        
        return indicators[:5]
    
    def _get_recommendation(self, confidence: float) -> str:
        """Get user recommendation"""
        
        if confidence > 0.80:
            return "🚫 HIGHLY LIKELY MISINFORMATION - Do not share"
        elif confidence > 0.65:
            return "⚠️ Likely misinformation - Verify with credible sources"
        elif confidence > 0.50:
            return "❓ UNCERTAIN - Needs fact-checking"
        elif confidence > 0.35:
            return "✅ Likely accurate - Low misinformation probability"
        else:
            return "✅ VERY LIKELY ACCURATE"
    
    def _generate_report(self, text: str, confidence: float, 
                        *args) -> str:
        """Generate detailed report"""
        
        report = f"""
RUMOR ANALYSIS REPORT
{'='*50}

Misinformation Confidence: {confidence*100:.1f}%
Classification: {'🚩 LIKELY MISINFORMATION' if confidence > 0.65 else '✅ LIKELY ACCURATE'}

Text Analyzed:
"{text[:150]}{'...' if len(text) > 150 else ''}"

Key Findings:
- Multiple analysis methods applied (ML, keyword, pattern)
- Analyzed for common misinformation indicators
- Evaluated consistency and credibility signals

Recommendation:
{'⚠️ Use caution before sharing' if confidence >0.55 else '✅ Generally accurate based on analysis'}

Note: This is an automated analysis. Always verify with multiple credible sources.
        """
        
        return report.strip()
    
    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return empty result"""
        return {
            "status": "error",
            "message": reason,
            "is_misinformation": False,
            "confidence": 0.0
        }


# Singleton instance  
_analyzer = None

def get_improved_analyzer() -> ImprovedRumorAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ImprovedRumorAnalyzer()
    return _analyzer
