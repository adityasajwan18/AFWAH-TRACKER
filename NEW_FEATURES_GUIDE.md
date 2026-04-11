# AFWAH Tracker - New Features Implementation Guide

**Implemented: April 2026**

This document covers the three major features added to the AFWAH Tracker system.

---

## 📡 Feature 1: Live Social Media Integration

### Overview
Real-time monitoring and analysis of posts from Twitter/X and Reddit directly through the API.

### Files Added
```
backend/integrations/
├── social_media_client.py      # Twitter & Reddit API clients
├── realtime_monitor.py         # Monitoring service & scheduler
└── __init__.py
backend/api/routes/
└── social_media.py             # API endpoints
```

### API Endpoints

#### Configuration
```
POST /api/social/configure/twitter
Body: {"bearer_token": "your_token"}

POST /api/social/configure/reddit
Body: {
  "client_id": "id",
  "client_secret": "secret",
  "user_agent": "optional_user_agent"
}
```

#### Monitoring Control
```
POST /api/social/monitor/start
Body: {
  "platform": "twitter" | "reddit",
  "query": "search keywords",
  "poll_interval": 60  # seconds
}
Returns: {"task_id": "twitter_123456..."}

POST /api/social/monitor/pause/{task_id}
POST /api/social/monitor/resume/{task_id}
POST /api/social/monitor/stop/{task_id}
GET /api/social/monitor/status/{task_id}
GET /api/social/monitor/tasks  # List all active tasks
```

#### Retrieve Monitored Posts
```
GET /api/social/posts?platform=twitter&limit=100
GET /api/social/posts?query=covid&limit=50
```

#### One-time Search
```
GET /api/social/search/twitter?query=vaccine&limit=100
GET /api/social/search/reddit?query=covid&limit=50
GET /api/social/trending/twitter?limit=50
GET /api/social/trending/reddit?limit=50
```

#### Analysis
```
POST /api/social/analyze
Body: {
  "platform": "twitter",
  "post_id": "123",
  "content": "tweet text",
  "author": "username"
}
```

### Usage Example

```python
import aiohttp
import asyncio

async def monitor_misinformation():
    # 1. Configure Twitter
    async with aiohttp.ClientSession() as session:
        await session.post(
            'http://localhost:8000/api/social/configure/twitter',
            json={'bearer_token': 'YOUR_BEARER_TOKEN'}
        )
    
    # 2. Start monitoring
    async with aiohttp.ClientSession() as session:
        resp = await session.post(
            'http://localhost:8000/api/social/monitor/start',
            json={
                'platform': 'twitter',
                'query': 'vaccine safety',
                'poll_interval': 60
            }
        )
        data = await resp.json()
        task_id = data['task_id']
    
    # 3. Get monitored posts
    async with aiohttp.ClientSession() as session:
        resp = await session.get(
            f'http://localhost:8000/api/social/posts',
            params={'platform': 'twitter'}
        )
        posts = await resp.json()
        
        for post in posts:
            print(f"{post['author']}: {post['content']}")
    
    # 4. Stop monitoring
    async with aiohttp.ClientSession() as session:
        await session.post(
            f'http://localhost:8000/api/social/monitor/stop/{task_id}'
        )

asyncio.run(monitor_misinformation())
```

### Setup Requirements

1. **Twitter API Access**
   - Go to https://developer.twitter.com/en/portal/dashboard
   - Create an app and generate Bearer token
   - Requires academic or enterprise access for `tweets/search/recent` endpoint

2. **Reddit API Access**
   - Go to https://www.reddit.com/prefs/apps
   - Create an app to get client ID and client secret
   - Free/basic access is sufficient

3. **Dependencies**
   ```bash
   pip install aiohttp tweepy praw
   ```

### Key Features

✅ **Real-time monitoring** - Continuous polling of social media streams  
✅ **Multiple platforms** - Twitter and Reddit support  
✅ **Integrated analysis** - Automatically analyzes posts for misinformation  
✅ **Persistent storage** - SQLite database of monitored posts  
✅ **Callback system** - Register custom callbacks for each detected post  
✅ **Task management** - Pause, resume, and stop monitoring tasks  

---

## 🔍 Feature 2: Browser Extension

### Overview
Real-time fact-checking overlay for social media posts with one-click analysis.

### Files Added
```
browser_extension/
├── manifest.json            # Extension configuration
├── popup.html              # Main UI
├── popup.js                # Popup logic & settings
├── content.js              # Injected content scripts
├── background.js           # Service worker
├── styles.css              # All styling
└── README.md               # Full documentation
```

### Installation

1. **Clone the extension files**
   ```bash
   cd browser_extension
   ```

2. **Chrome/Edge Installation**
   - Navigate to `chrome://extensions/` or `edge://extensions/`
   - Enable "Developer Mode" (top right)
   - Click "Load unpacked"
   - Select the `browser_extension` folder

3. **Firefox Installation** (requires some adjustments)
   - Go to `about:debugging`
   - Click "Load Temporary Add-on"
   - Select any file in the extension folder

### Features

**Manual Analysis**
- Click extension icon
- Paste any text in "Analyzer" tab
- Click "🔍 Analyze"
- View results with confidence score and recommendations

**Automatic Page Checking**
- Enable "Auto-check posts on page" in Settings
- Extension automatically analyzes posts as you scroll
- High-risk posts get colored badges

**Inline Indicators**
- 🟢 Green badge (Low Risk) - Likely accurate
- 🟡 Orange badge (Medium Risk) - Verify with sources
- 🔴 Red badge (High Risk) - Likely misinformation

**Configurable Settings**
- API URL: Point to your backend
- Confidence Threshold: 50%, 65% (default), 80%, 90%
- Auto-check: Toggle automatic analysis

### Configuration

In extension popup → Settings tab:

```
API URL: http://localhost:8000
Confidence Threshold: 65%
Auto-check posts: ✓ Enabled
```

### Supported Sites

✅ Twitter.com / X.com  
✅ Facebook.com  
✅ Reddit.com  
✅ TikTok.com (limited)  

### Development

**Add new platform support:**

1. Update `PLATFORM_SELECTORS` in `content.js`:
   ```javascript
   newsite: {
     postContainer: '[your-selector]',
     postText: '[text-selector]',
     author: '[author-selector]'
   }
   ```

2. Add host permissions in `manifest.json`:
   ```json
   "host_permissions": [
     "https://yoursite.com/*"
   ]
   ```

3. Update content script injection list

---

## 🎬 Feature 3: Deepfake/Synthetic Media Detection

### Overview
Comprehensive detection of AI-generated and manipulated videos, audio, and images.

### Files Added
```
backend/ml/
└── deepfake_detector.py    # Deepfake & synthetic audio detection
backend/api/routes/
└── media.py                # Media analysis endpoints
```

### API Endpoints

#### Video Analysis
```
POST /api/analyze-video
Content-Type: multipart/form-data
File: video file (.mp4, .avi, .mov, .mkv, etc.)

Response:
{
  "is_deepfake": true,
  "confidence": 0.82,
  "frames_analyzed": 45,
  "anomalies_detected": 12,
  "anomalies": [
    {"frame": 5, "type": "ai_generation", "score": 0.85},
    ...
  ],
  "scores": {
    "avg_frame_ai_score": 0.78,
    "temporal_consistency": 0.75,
    "frequency_artifacts": 0.68
  },
  "details": "🚨 VERY HIGH RISK...",
  "analyzed_at": "2026-04-12T10:30:00Z"
}
```

#### Audio Analysis
```
POST /api/analyze-audio
Content-Type: multipart/form-data
File: audio file (.mp3, .wav, .flac, .m4a, etc.)

Response:
{
  "is_synthetic": true,
  "confidence": 0.71,
  "duration_seconds": 45.3,
  "methods": {
    "spectral_analysis": 0.68,
    "mfcc_consistency": 0.75,
    "zero_crossing": 0.62,
    "pitch_analysis": 0.78
  },
  "details": "⚠️ LIKELY SYNTHETIC...",
  "analyzed_at": "2026-04-12T10:30:00Z"
}
```

#### Combined Media Analysis
```
POST /api/analyze-media
Content-Type: multipart/form-data
File: video file (analyzes both video & audio)

Response includes both video and audio results with combined verdict
```

#### URL-based Analysis
```
POST /api/analyze-media-url?url=https://...&media_type=video|audio

Useful for analyzing social media videos without direct upload
```

### Detection Methods

#### Video Analysis

1. **Frame-by-frame AI Detection**
   - Uses existing AI image detector on extracted frames
   - Identifies AI-generated faces and artifacts
   - Detects facial feature inconsistencies

2. **Temporal Consistency Analysis**
   - Analyzes variance of AI scores across frames
   - Deepfakes often have temporal discontinuities
   - Real videos have more consistent faces

3. **Frequency Domain Analysis**
   - FFT analysis of frame data
   - Detects compression and encoding artifacts
   - Identifies unnatural frequency patterns

4. **Optical Flow Analysis**
   - Tracks motion consistency
   - Detects unnatural movement patterns
   - Identifies generation artifacts

#### Audio Analysis

1. **Spectral Analysis**
   - Analyzes frequency spectrum characteristics
   - Synthetic speech has more uniform spectrum
   - Detects TTS signatures

2. **MFCC Consistency**
   - Mel-frequency cepstral coefficients analysis
   - Measures consistency across time
   - AI voices often have unusual patterns

3. **Zero Crossing Rate**
   - Analyzes speech waveform characteristics
   - Synthetic speech has different ZCR patterns
   - Detects unusual voicing characteristics

4. **Pitch Analysis**
   - Fundamental frequency extraction
   - Synthetic voices have more consistent pitch
   - Real voices have natural pitch variation

### Usage Examples

```bash
# Analyze video file
curl -X POST "http://localhost:8000/api/analyze-video" \
  -F "file=@video.mp4"

# Analyze audio file
curl -X POST "http://localhost:8000/api/analyze-audio" \
  -F "file=@audio.wav"

# Analyze full media (video + audio)
curl -X POST "http://localhost:8000/api/analyze-media" \
  -F "file=@video.mp4"

# Analyze from URL
curl -X POST "http://localhost:8000/api/analyze-media-url" \
  -d "url=https://twitter.com/video.mp4" \
  -d "media_type=video"
```

### Python Integration

```python
import requests

# Analyze video
with open('video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze-video',
        files={'file': f}
    )
    result = response.json()
    
    print(f"Is deepfake: {result['is_deepfake']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Frames analyzed: {result['frames_analyzed']}")
    print(f"Anomalies: {result['anomalies_detected']}")
    print(result['details'])

# Analyze audio
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/analyze-audio',
        files={'file': f}
    )
    result = response.json()
    
    print(f"Is synthetic: {result['is_synthetic']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(result['methods'])
```

### Installation

1. **Install additional dependencies**
   ```bash
   pip install opencv-python librosa soundfile
   ```

2. **Verify installation**
   ```bash
   python -c "import cv2; import librosa; print('OK')"
   ```

### Performance Notes

- **Video Processing**: ~2-5 seconds per minute of video
- **Audio Processing**: Real-time (faster than audio duration)
- **Frame Sampling**: Extracts every 10th frame (configurable)
- **Max Frames Analyzed**: 50 frames per video

### Size Limits

- **Video**: 100 MB max
- **Audio**: 50 MB max
- **Recommended**: Under 50 MB for faster processing

### Confidence Interpretation

| Score | Interpretation | Action |
|-------|----------------|--------|
| > 0.85 | Very High Risk | 🚨 Do not share |
| 0.65-0.85 | High Risk | ⚠️ Verify sources |
| 0.5-0.65 | Medium Risk | ❓ Investigate |
| < 0.5 | Low Risk | ✅ Likely authentic |

---

## 📦 Dependencies Update

Added to `requirements.txt`:

```
aiohttp==3.9.0                  # Async HTTP for APIs
tweepy==4.14.0                  # Twitter API (optional)
praw==7.7.1                     # Reddit API (optional)
opencv-python==4.9.0.80         # Video processing
moviepy==1.0.3                  # Video manipulation
librosa==0.10.0                 # Audio analysis
soundfile==0.12.1               # Audio I/O
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Social Media APIs (Optional)
```bash
# Get Twitter Bearer token: https://developer.twitter.com
# Get Reddit credentials: https://www.reddit.com/prefs/apps
```

### 3. Start Backend
```bash
uvicorn main:app --reload --port 8000
```

### 4. Install Browser Extension
- Navigate to `browser_extension` folder
- Follow installation instructions in `browser_extension/README.md`

### 5. Start Using
- Visit social media sites and see real-time fact-checking
- Use extension popup for manual analysis
- Use API endpoints for programmatic access

---

## 🔧 Configuration

### main.py
Routes are registered:
```python
from backend.api.routes import social_media, media
app.include_router(social_media.router, prefix="/api", tags=["Social Media"])
app.include_router(media.router, prefix="/api", tags=["Media Analysis"])
```

### .env (Optional)
```
TWITTER_BEARER_TOKEN=your_token
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

---

## 📊 API Documentation

Auto-generated Swagger docs available at:
```
http://localhost:8000/docs
```

ReDoc documentation:
```
http://localhost:8000/redoc
```

All new endpoints are documented with:
- Request/response schemas
- Example payloads
- Error handling
- Rate limits (if applicable)

---

## 🐛 Troubleshooting

### Social Media Integration

**"Platform not registered"**
- Call `/api/social/configure/twitter` or `/configure/reddit` first

**"API error"**
- Check if API credentials are valid
- Verify network connectivity
- Check rate limits

### Browser Extension

**"Analysis failed"**
- Ensure backend is running: `uvicorn main:app --reload`
- Check API URL in settings
- Try manual analysis to verify API works

**No overlays appearing**
- Enable "Auto-check posts" in settings
- Refresh the page
- Check browser console for errors

### Deepfake Detection

**"librosa not installed"**
- Run: `pip install librosa soundfile`

**Analysis taking too long**
- Try smaller file size
- Adjust frame sampling rate in code

**Out of memory**
- Reduce MAX_FRAMES in deepfake_detector.py
- Process smaller videos

---

## 📝 Summary

✅ **Live Social Media Integration** - Real-time Twitter & Reddit monitoring  
✅ **Browser Extension** - Inline fact-checking on social media  
✅ **Deepfake Detection** - Video & audio synthetic media detection  

All features are integrated into the existing AFWAH backend and ready for production use.

For detailed documentation on each feature, see:
- `browser_extension/README.md` - Extension setup & usage
- Code comments in respective files

---

**Last Updated**: April 12, 2026  
**Status**: Production Ready ✅
