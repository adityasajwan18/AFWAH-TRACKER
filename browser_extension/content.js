// ============================================================
// content.js - Content Script
// Injected into social media pages for real-time checking
// ============================================================

let settings = {
  apiUrl: 'http://localhost:8000',
  autoCheck: true,
  confidenceThreshold: 0.65
};

// Load settings
chrome.storage.sync.get(
  ['apiUrl', 'autoCheck', 'confidenceThreshold'],
  (items) => {
    if (items.apiUrl) settings.apiUrl = items.apiUrl;
    if (items.autoCheck !== undefined) settings.autoCheck = items.autoCheck;
    if (items.confidenceThreshold) settings.confidenceThreshold = items.confidenceThreshold;
    
    if (settings.autoCheck) {
      initializePageMonitoring();
    }
  }
);

// ── Detect current platform ─────────────────────────────────
function getCurrentPlatform() {
  const hostname = window.location.hostname;
  if (hostname.includes('twitter.com') || hostname.includes('x.com')) {
    return 'twitter';
  } else if (hostname.includes('facebook.com')) {
    return 'facebook';
  } else if (hostname.includes('reddit.com')) {
    return 'reddit';
  } else if (hostname.includes('tiktok.com')) {
    return 'tiktok';
  }
  return 'unknown';
}

// ── Platform-specific selectors ─────────────────────────────
const PLATFORM_SELECTORS = {
  twitter: {
    postContainer: '[data-testid="tweet"]',
    postText: '[data-testid="tweetText"]',
    author: '[data-testid="User-Name"]'
  },
  facebook: {
    postContainer: '[data-testid="post"]',
    postText: '.x1iorvi4',
    author: 'h3'
  },
  reddit: {
    postContainer: '[data-testid="post-container"]',
    postText: 'h3',
    author: 'a[data-testid="comment_author_link"]'
  }
};

// ── Initialize page monitoring ──────────────────────────────
function initializePageMonitoring() {
  const platform = getCurrentPlatform();
  
  if (platform === 'unknown') return;
  
  console.log(`[AFWAH] Monitoring ${platform} posts for misinformation`);
  
  // Monitor for new posts using MutationObserver
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.addedNodes.length) {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) { // Element node
            const posts = node.querySelectorAll(PLATFORM_SELECTORS[platform].postContainer);
            posts.forEach(post => {
              if (!post.dataset.afwahChecked) {
                analyzePost(post, platform);
              }
            });
          }
        });
      }
    });
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: false
  });
  
  // Check existing posts
  const existingPosts = document.querySelectorAll(PLATFORM_SELECTORS[platform].postContainer);
  existingPosts.forEach(post => analyzePost(post, platform));
}

// ── Analyze individual post ─────────────────────────────────
async function analyzePost(postElement, platform) {
  try {
    // Mark as checked
    postElement.dataset.afwahChecked = 'true';
    
    // Extract post text
    const textElement = postElement.querySelector(PLATFORM_SELECTORS[platform].postText);
    if (!textElement) return;
    
    const postText = textElement.textContent.trim();
    if (!postText || postText.length < 10) return;
    
    // Analyze
    const response = await fetch(`${settings.apiUrl}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: postText })
    });
    
    if (!response.ok) throw new Error('Analysis failed');
    
    const result = await response.json();
    
    // Only show overlay if confidence exceeds threshold
    if (result.confidence >= settings.confidenceThreshold) {
      injectOverlay(postElement, result, platform);
    }
    
  } catch (error) {
    console.error('[AFWAH] Analysis error:', error);
  }
}

// ── Inject analysis overlay ─────────────────────────────────
function injectOverlay(postElement, result, platform) {
  // Check if overlay already exists
  if (postElement.querySelector('.afwah-overlay')) return;
  
  const confidence = result.confidence || 0;
  const isMisinformation = result.is_misinformation || false;
  
  // Determine severity
  let severity = 'low';
  let icon = '✅';
  let color = '#4CAF50';
  let message = 'Low Risk';
  
  if (confidence > 0.85) {
    severity = 'high';
    icon = '🚩';
    color = '#D32F2F';
    message = 'High Risk';
  } else if (confidence > 0.65) {
    severity = 'medium';
    icon = '⚠️';
    color = '#FF9800';
    message = 'Medium Risk';
  }
  
  // Create overlay element
  const overlay = document.createElement('div');
  overlay.className = `afwah-overlay afwah-${severity}`;
  overlay.innerHTML = `
    <div class="afwah-badge" style="border-left: 4px solid ${color}">
      <div class="afwah-header">
        <span class="afwah-icon">${icon}</span>
        <span class="afwah-message">${message}</span>
        <span class="afwah-confidence">${(confidence * 100).toFixed(0)}%</span>
      </div>
      <div class="afwah-actions">
        <button class="afwah-btn" onclick="alert('Full analysis in AFWAH popup!')">
          More Info
        </button>
      </div>
    </div>
  `;
  
  // Insert overlay
  postElement.style.position = 'relative';
  postElement.insertBefore(overlay, postElement.firstChild);
  
  // Add visual indicator to post
  postElement.style.borderLeft = `3px solid ${color}`;
  postElement.style.paddingLeft = '10px';
}

// ── Listen for messages from popup ──────────────────────────
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'analyzeSelectedText') {
    const selectedText = window.getSelection().toString();
    if (selectedText) {
      chrome.runtime.sendMessage({
        action: 'textSelected',
        text: selectedText
      });
    }
  }
});

console.log('[AFWAH] Content script loaded on', getCurrentPlatform());
