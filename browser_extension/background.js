// ============================================================
// background.js - Service Worker
// Background tasks and event handlers
// ============================================================

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    chrome.tabs.create({ url: 'chrome-extension://' + chrome.runtime.id + '/popup.html' });
    console.log('[AFWAH] Extension installed');
  } else if (details.reason === 'update') {
    console.log('[AFWAH] Extension updated');
  }
});

// Listen for messages from content scripts or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'textSelected') {
    // Store selected text for popup analysis
    chrome.storage.session.set({ selectedText: request.text });
  }
  
  if (request.action === 'reportMisinformation') {
    // Send report to backend
    fetch('http://localhost:8000/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        platform: request.platform,
        post_id: request.postId,
        text: request.text,
        confidence: request.confidence
      })
    }).catch(err => console.error('[AFWAH] Report error:', err));
  }
});

// Monitor tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete') {
    // Inject content script when page fully loads
    const supportedHosts = [
      'twitter.com',
      'x.com',
      'facebook.com',
      'reddit.com',
      'tiktok.com'
    ];
    
    const isSupportedHost = supportedHosts.some(host => 
      tab.url && tab.url.includes(host)
    );
    
    if (isSupportedHost) {
      chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ['content.js']
      }).catch(err => {
        // Script may already be injected
        console.debug('[AFWAH] Script injection:', err.message);
      });
    }
  }
});

// Badge to show extension is active
chrome.action.setBadgeBackgroundColor({ color: '#FF6B6B' });
chrome.action.setBadgeText({ text: '✓' });

console.log('[AFWAH] Background service worker loaded');
