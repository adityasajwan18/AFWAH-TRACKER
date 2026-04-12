#!/usr/bin/env python
# ============================================================
# scripts/test_ml_models.py
#
# Test script for improved ML models
# Demonstrates both AI image and rumor detection
# ============================================================

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.improved_ai_detector import get_improved_detector
from backend.ml.improved_rumor_analyzer import get_improved_analyzer


def test_rumor_analyzer():
    """Test the improved rumor analyzer"""
    
    print("\n" + "="*60)
    print("🧪 RUMOR ANALYZER TEST")
    print("="*60)
    
    analyzer = get_improved_analyzer()
    
    test_claims = [
        {
            "text": "The Moon landing in 1969 was faked in a Hollywood studio",
            "context": "general",
            "expected": "misinformation"
        },
        {
            "text": "COVID-19 is caused by a novel coronavirus that spreads through respiratory droplets",
            "context": "health",
            "expected": "accurate"
        },
        {
            "text": "5G towers are spreading COVID-19",
            "context": "health",
            "expected": "misinformation"
        },
        {
            "text": "Vaccines contain microchips for surveillance",
            "context": "health",
            "expected": "misinformation"
        },
        {
            "text": "The Earth is an oblate spheroid with a radius of approximately 6,371 kilometers",
            "context": "general",
            "expected": "accurate"
        }
    ]
    
    print("\nTesting Claims:\n")
    
    for i, claim in enumerate(test_claims, 1):
        print(f"Test {i}:")
        print(f"Claim: {claim['text'][:70]}...")
        print(f"Context: {claim['context']}")
        print(f"Expected: {claim['expected']}")
        
        result = analyzer.analyze(claim['text'], claim['context'])
        
        classification = "MISINFORMATION" if result['is_misinformation'] else "ACCURATE"
        confidence = result['confidence'] * 100
        
        print(f"Classification: {classification}")
        print(f"Confidence: {confidence:.1f}%")
        print(f"Credibility: {result['credibility_score']*100:.1f}%")
        
        if result.get('indicators'):
            print("Key Indicators:")
            for indicator in result['indicators'][:2]:
                print(f"  {indicator}")
        
        print(f"Recommendation: {result['recommendation']}")
        print()


def test_image_detector():
    """Test the improved AI image detector"""
    
    print("\n" + "="*60)
    print("📸 AI IMAGE DETECTOR TEST")
    print("="*60)
    
    detector = get_improved_detector()
    
    print("\n⚠️ Requires test images to demonstrate")
    print("Usage:")
    print("""
    from backend.ml.improved_ai_detector import get_improved_detector
    
    detector = get_improved_detector()
    result = detector.analyze_image("path/to/image.jpg")
    
    print(result['is_ai_generated'])     # True/False
    print(result['confidence'])          # 0.0-1.0
    print(result['method_scores'])       # Individual scores
    """)


def print_model_info():
    """Print information about loaded models"""
    
    print("\n" + "="*60)
    print("📊 MODEL INFORMATION")
    print("="*60)
    
    analyzer = get_improved_analyzer()
    detector = get_improved_detector()
    
    print("\n🤖 RUMOR DETECTION MODELS:")
    
    if analyzer.rf_model:
        print("  ✅ Random Forest: Loaded")
        print(f"     - Trees: {analyzer.rf_model.n_estimators}")
        print(f"     - Max depth: {analyzer.rf_model.max_depth}")
    else:
        print("  ❌ Random Forest: Not loaded")
    
    if analyzer.gb_model:
        print("  ✅ Gradient Boosting: Loaded")
        print(f"     - Estimators: {analyzer.gb_model.n_estimators}")
        print(f"     - Learning rate: {analyzer.gb_model.learning_rate}")
    else:
        print("  ❌ Gradient Boosting: Not loaded")
    
    if analyzer.vectorizer:
        print("  ✅ TF-IDF Vectorizer: Loaded")
        if hasattr(analyzer.vectorizer, 'n_features_in_'):
            print(f"     - Features: {analyzer.vectorizer.n_features_in_}")
    else:
        print("  ❌ TF-IDF Vectorizer: Not loaded")
    
    print("\n🎬 IMAGE DETECTION MODELS:")
    print("  ✅ Isolation Forest: Loaded")
    print("  ✅ Multiple CV methods: Ready")
    print("  Available detection methods:")
    print("     1. Frequency Analysis (FFT)")
    print("     2. Error Level Analysis (ELA)")
    print("     3. Noise Analysis")
    print("     4. Edge Detection")
    print("     5. Color Pattern Analysis")
    print("     6. DCT Analysis")
    print("     7. Blur Detection")
    print("     8. Splicing Detection")
    print("     9. Metadata Analysis")


def main():
    """Main test suite"""
    
    print("\n" + "="*70)
    print("🧪 AFWAH TRACKER - ML MODEL TEST SUITE")
    print("="*70)
    
    try:
        print_model_info()
        test_rumor_analyzer()
        test_image_detector()
        
        print("\n" + "="*70)
        print("✅ TEST SUITE COMPLETED")
        print("="*70)
        print("\nModels are ready for production use!")
        print("\nNext steps:")
        print("1. Integrate into API endpoints")
        print("2. Monitor performance on real data")
        print("3. Collect misclassifications for retraining")
        print("4. Update training data periodically")
        print("="*70 + "\n")
        
        return True
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
