# AFWAH Browser Extension

Quick fact-checking overlays for social media posts across Twitter, Facebook, Reddit, and TikTok.

## Features

✅ **Real-time Post Analysis** - Automatically checks posts for misinformation  
✅ **Visual Indicators** - Color-coded severity badges (🟢 Low / 🟡 Medium / 🔴 High)  
✅ **Popup Analyzer** - Manual analysis of any text  
✅ **Auto-check Toggle** - Enable/disable automatic checking  
✅ **Configurable Threshold** - Set confidence level (50% - 90%)  
✅ **Custom API URL** - Point to your own AFWAH backend

## Supported Platforms

- Twitter / X
- Facebook  
- Reddit
- TikTok (limited support)

## Installation

### Development Mode

1. Clone the repository and navigate to the extension folder:
   ```bash
   cd browser_extension
   ```

2. Open Chrome/Edge and go to `chrome://extensions/` (or `edge://extensions/`)

3. Enable "Developer Mode" (top right)

4. Click "Load unpacked" and select the `browser_extension` folder

5. The extension is now installed! Look for the AFWAH icon in your toolbar

### Production Build

```bash
# Zip the extension for Chrome Web Store submission
zip -r afwah-extension.zip . -x "*.git*" "*.md"
```

## Configuration

### API Setup

In the extension popup, Settings tab:

1. **API URL**: Point to your AFWAH backend (default: `http://localhost:8000`)
2. **Auto-check Posts**: Toggle automatic analysis on social media pages
3. **Confidence Threshold**: 
   - Low (50%): More detections, more false positives
   - Medium (65%): Balanced (default)
   - High (80%): Fewer detections, more accurate
   - Very High (90%): Most conservative

### Backend Requirements

Ensure your AFWAH backend is running with:
- `/api/analyze` endpoint (POST)
- CORS properly configured for extension origin
- Misinformation classification model loaded

## Usage

### Quick Analysis

1. Click the AFWAH icon in your toolbar
2. Paste text in the "Analyzer" tab
3. Click "🔍 Analyze"
4. View results with confidence score and recommendations

### On Social Media

1. Enable "Auto-check posts on page" in Settings
2. The extension automatically analyzes posts as you scroll
3. High-risk posts get a colored badge with severity indicator
4. Click "More Info" on any badge for detailed analysis

### Sharing Results

- **Copy Text**: Copies the analyzed text to clipboard
- **Share**: Creates shareable link with analysis
- **Report**: Submit feedback to help improve detection

## Architecture

```
browser_extension/
├── manifest.json          # Extension configuration
├── popup.html             # Main UI
├── popup.js               # Popup logic & API calls
├── content.js             # Injected into social media pages
├── background.js          # Service worker
├── styles.css             # All styling
└── README.md              # This file
```

### Main Components

**manifest.json**
- Declares permissions and content scripts
- Specifies host permissions for social media sites
- Defines action popup and icons

**popup.html/popup.js**
- Main UI for the extension
- Settings management
- Manual text analysis interface

**content.js**
- Injected into social media page DOM
- Monitors for new posts using MutationObserver
- Displays inline badges with analysis results
- Platform-specific selectors for each site

**background.js**
- Service worker handling messages
- Reports misinformation to backend
- Handles tab updates and script injection

**styles.css**
- Popup UI styling
- Content script overlay styling
- Animations and responsive design

## API Integration

### The extension communicates with the backend via:

```
POST /api/analyze
Content-Type: application/json

{
  "text": "claim to analyze"
}
```

**Response:**
```json
{
  "is_misinformation": true,
  "confidence": 0.87,
  "classification": "misinformation",
  "status": "analyzed"
}
```

## Troubleshooting

### Extension not analyzing posts?

1. ✅ Check if "Auto-check posts on page" is enabled in Settings
2. ✅ Verify API URL is correct (usually `http://localhost:8000`)
3. ✅ Ensure backend is running: `uvicorn main:app --reload`
4. ✅ Open Chrome DevTools (F12) → Console → Check for errors
5. ✅ Try manual analysis in popup to verify API connectivity

### No overlay appearing?

- Check console for errors: Right-click page → Inspect → Console tab
- Verify confidence threshold setting
- Try enabling "Auto-check" in Settings tab

### "API error" message?

- ✅ Backend not running: `uvicorn main:app --reload --port 8000`
- ✅ Wrong API URL in settings
- ✅ CORS issues: Check backend CORS configuration
- ✅ Try localhost vs 127.0.0.1

### Posts take too long to analyze?

- ✅ Check backend performance
- ✅ Reduce polling frequency in content.js
- ✅ Enable model caching in backend (`backend/ml/model_optimizer.py`)

## Security Notes

⚠️ The extension sends post text to your configured backend API. Ensure:

- ✅ Backend is on HTTPS in production
- ✅ No sensitive/private posts are being analyzed
- ✅ API authentication is enabled if backend is public
- ✅ Extension only works on social media sites

## Development

### Adding a new platform

1. Add selectors to `PLATFORM_SELECTORS` in content.js:
   ```javascript
   newplatform: {
     postContainer: '[selector]',
     postText: '[selector]',
     author: '[selector]'
   }
   ```

2. Update host_permissions in manifest.json

3. Update initializePageMonitoring() platform detection

### Debugging

Enable debug logging:
```javascript
// In any script
console.log('[AFWAH] Message here');
```

View logs in Chrome DevTools → Extensions tab → Background → Logs

## License

MIT - See LICENSE file

## Contributing

Issues and PRs welcome! Please test with the AFWAH backend before submitting.
