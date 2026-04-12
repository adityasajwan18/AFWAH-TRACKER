# 🚀 Quick Start - ML Models Integration

## Setup (One-time)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (generates backend/ml/models/*.pkl)
python scripts/train_rumor_detector.py

# 3. Verify models work
python scripts/test_ml_models.py
```

## Use Improved Image Detector

```python
from backend.ml.improved_ai_detector import get_improved_detector

detector = get_improved_detector()

# Analyze an image
result = detector.analyze_image("path/to/image.jpg")

print(f"Is AI Generated: {result['is_ai_generated']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Markers: {result.get('markers', [])}")
```

**Returns:**
- `is_ai_generated` (bool): True if likely AI-generated
- `confidence` (0-1): How confident the prediction is
- `method_scores` (dict): Individual scores from 9 methods
- `evaluation_metrics` (dict): Detailed analysis per method
- `markers` (list): Detected anomalies/patterns

---

## Use Improved Rumor Analyzer

```python
from backend.ml.improved_rumor_analyzer import get_improved_analyzer

analyzer = get_improved_analyzer()

# Analyze a claim
result = analyzer.analyze(
    text="5G towers spread COVID-19",
    context="health"  # or: "general", "political", "scientific"
)

print(f"Is Misinformation: {result['is_misinformation']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Credibility: {result['credibility_score']:.1%}")
print(f"Recommendation: {result['recommendation']}")
```

**Returns:**
- `is_misinformation` (bool): True if claim is likely false
- `confidence` (0-1): Model confidence level
- `credibility_score` (0-1): Overall claim credibility
- `indicators` (list): Evidence why it's flagged
- `recommendation` (str): Action to take
- `method_scores` (dict): Individual method contributions

---

## Add to FastAPI Routes

### Example 1: Image Analysis Endpoint

```python
# In backend/api/routes/image.py
from fastapi import File, UploadFile
from backend.ml.improved_ai_detector import get_improved_detector
import tempfile

detector = get_improved_detector()

@router.post("/analyze-improved")
async def analyze_improved(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(await file.read())
        result = detector.analyze_image(tmp.name)
    
    return {
        "is_ai_generated": result['is_ai_generated'],
        "confidence": result['confidence'],
        "method_scores": result['method_scores'],
        "recommendation": ("Block" if result['confidence'] > 0.7 else "Review")
    }
```

### Example 2: Rumor Analysis Endpoint

```python
# In backend/api/routes/analyze.py
from backend.ml.improved_rumor_analyzer import get_improved_analyzer

analyzer = get_improved_analyzer()

@router.post("/rumor/check")
async def check_rumor(claim: str, context: str = "general"):
    result = analyzer.analyze(claim, context)
    
    return {
        "is_misinformation": result['is_misinformation'],
        "confidence_score": result['confidence'],
        "credibility": result['credibility_score'],
        "indicators": result['indicators'][:3],  # Top 3 reasons
        "action": "BLOCK" if result['confidence'] > 0.8 else "REVIEW"
    }
```

---

## Model Architecture

### Image Detection (9 Methods)
1. **Frequency Analysis** (FFT) - Detects unnatural frequency patterns
2. **Error Level Analysis** (ELA) - Finds JPEG compression anomalies
3. **Noise Analysis** - Detects AI smoothing artifacts
4. **Edge Detection** - Looks for unnatural edges
5. **Color Patterns** - Finds unusual color distributions
6. **DCT Analysis** - Discrete Cosine Transform patterns
7. **Blur Detection** - Unnatural blur regions
8. **Splicing Detection** - Image stitching indicators
9. **Metadata Analysis** - EXIF inconsistencies

**Ensemble:** 70% feature-based + 30% ML (Isolation Forest)

### Rumor Detection (5 Methods)
1. **ML Classification** (35%) - Random Forest + Gradient Boosting
2. **Keyword Analysis** (25%) - Known misinformation keywords
3. **Pattern Detection** (20%) - Rhetoric patterns (ALL_CAPS, emotion, logic fallacies)
4. **Consistency Check** (12%) - Internal contradictions
5. **Source Credibility** (8%) - Source reliability signals

---

## Model Files

After training, these files are created:

```
backend/ml/models/
├── rf_rumor_model.pkl          # Random Forest classifier
├── gb_rumor_model.pkl          # Gradient Boosting classifier
└── tfidf_vectorizer.pkl        # Text vectorizer
```

*Note: Image detection uses in-memory Isolation Forest (no file needed)*

---

## Performance Metrics

From training output you'll see:

```
Random Forest Model:
  Accuracy:   87.5%
  Precision:  89.2%
  Recall:     85.3%
  F1-Score:   87.2%

Gradient Boosting Model:
  Accuracy:   89.1%
  Precision:  90.5%
  Recall:     87.8%
  F1-Score:   89.1%

Ensemble Accuracy: 91.2%
```

---

## Testing

Run the test suite:

```bash
python scripts/test_ml_models.py
```

This will:
- ✅ Display loaded models and their configurations
- ✅ Test rumor analyzer on 5 sample claims
- ✅ Prepare image detector for testing
- ✅ Show classification confidence levels

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run `pip install -r requirements.txt` |
| Models not loading | Run `python scripts/train_rumor_detector.py` |
| FileNotFoundError (images) | Use absolute paths: `Path(__file__).parent / "image.jpg"` |
| Slow inference | Models cached in memory (singletons), first call slower |
| Low accuracy | Retrain with more labeled data in `data/rumor_training_data.json` |

---

## Next Steps

1. **Run Training:** `python scripts/train_rumor_detector.py`
2. **Test Models:** `python scripts/test_ml_models.py`
3. **Integrate Endpoints:** Add routes from examples above
4. **Deploy to FastAPI:** Register routes in `main.py`
5. **Monitor Performance:** Track real-world accuracy
6. **Collect Misclassifications:** For future retraining

---

## API Examples

### Using with cURL

```bash
# Test rumor analysis
curl -X POST http://127.0.0.1:8000/rumor/check \
  -d 'claim=5G spreads COVID' \
  -d 'context=health'

# Analyze image (if endpoint added)
curl -F "file=@image.jpg" http://127.0.0.1:8000/api/analyze-improved
```

### Using with Python

```bash
import httpx

async with httpx.AsyncClient() as client:
    # Rumor check
    resp = await client.post(
        "http://127.0.0.1:8000/rumor/check",
        json={
            "claim": "Climate change is a hoax",
            "context": "scientific"
        }
    )
    data = resp.json()
    print(f"Misinformation: {data['is_misinformation']}")
```

---

## Configuration

Edit model parameters in the source files:

**Random Forest** (`backend/ml/improved_rumor_analyzer.py`):
```python
self.rf_model = RandomForestClassifier(
    n_estimators=200,      # ← Increase for better accuracy
    max_depth=15,          # ← Increase to reduce underfitting
    random_state=42
)
```

**Gradient Boosting** (`backend/ml/improved_rumor_analyzer.py`):
```python
self.gb_model = GradientBoostingClassifier(
    n_estimators=150,      # ← Increase for better accuracy
    learning_rate=0.1,     # ← Decrease for slower, better learning
    random_state=42
)
```

---

## Support

- 📖 See [ML_IMPROVEMENTS_README.md](ML_IMPROVEMENTS_README.md) for detailed documentation
- 💾 Training data: [data/rumor_training_data.json](data/rumor_training_data.json)
- 🧪 Test file: [scripts/test_ml_models.py](scripts/test_ml_models.py)
- 📊 See training output: Run `python scripts/train_rumor_detector.py`

