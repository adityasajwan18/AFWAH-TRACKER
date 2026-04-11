// ============================================================
// popup.js - Browser Extension Popup Scripts
// Handles UI interactions and API calls
// ============================================================

const API_ENDPOINT = '/api/analyze';
let apiUrl = 'http://localhost:8000';

// ── DOM Elements ────────────────────────────────────────────
const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingDiv = document.getElementById('loading');
const resultsDiv = document.getElementById('results');
const errorDiv = document.getElementById('error');
const resultCard = document.getElementById('result-card');

const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

const apiUrlInput = document.getElementById('api-url');
const autoCheckCheckbox = document.getElementById('auto-check');
const confidenceThresholdSelect = document.getElementById('confidence-threshold');
const saveSettingsBtn = document.getElementById('save-settings-btn');
const settingsStatus = document.getElementById('settings-status');

// ── Initialize ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSettings();
  setupEventListeners();
});

function setupEventListeners() {
  // Tab switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabName = btn.dataset.tab;
      switchTab(tabName);
    });
  });
  
  // Analysis
  analyzeBtn.addEventListener('click', analyzText);
  textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      analyzText();
    }
  });
  
  // Settings
  saveSettingsBtn.addEventListener('click', saveSettings);
}

// ── Tab Management ─────────────────────────────────────────
function switchTab(tabName) {
  // Update buttons
  tabButtons.forEach(btn => {
    btn.classList.remove('active');
    if (btn.dataset.tab === tabName) {
      btn.classList.add('active');
    }
  });
  
  // Update content
  tabContents.forEach(content => {
    content.classList.remove('active');
    if (content.id === tabName) {
      content.classList.add('active');
    }
  });
}

// ── Analysis ────────────────────────────────────────────────
async function analyzText() {
  const text = textInput.value.trim();
  
  if (!text) {
    showError('Please enter some text to analyze');
    return;
  }
  
  try {
    showLoading(true);
    hideError();
    hideResults();
    
    const response = await fetch(`${apiUrl}${API_ENDPOINT}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    const result = await response.json();
    displayResults(result);
    
  } catch (error) {
    console.error('Analysis error:', error);
    showError(`Failed to analyze: ${error.message}`);
  } finally {
    showLoading(false);
  }
}

function displayResults(result) {
  hideResults();
  
  const confidence = result.confidence || 0;
  const isMisinformation = result.is_misinformation || false;
  const classification = result.classification || 'Unknown';
  
  // Determine severity color
  let severityColor = '#4CAF50'; // Green
  let severityLabel = '✅ Low Risk';
  
  if (confidence > 0.85) {
    severityColor = '#D32F2F'; // Red
    severityLabel = '🚩 High Risk';
  } else if (confidence > 0.65) {
    severityColor = '#FF9800'; // Orange
    severityLabel = '⚠️ Medium Risk';
  }
  
  resultCard.innerHTML = `
    <div class="severity-indicator" style="border-left: 4px solid ${severityColor}">
      <div class="severity-header">
        <span class="severity-label">${severityLabel}</span>
        <span class="confidence-badge">${(confidence * 100).toFixed(1)}% Confidence</span>
      </div>
      
      <div class="result-details">
        <div class="detail-row">
          <span class="detail-label">Classification:</span>
          <span class="detail-value">${classification}</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">Type:</span>
          <span class="detail-value">${isMisinformation ? 'Potential Misinformation' : 'Likely Accurate'}</span>
        </div>
        
        <div class="detail-row">
          <span class="detail-label">Status:</span>
          <span class="detail-value">${result.status || 'Analyzed'}</span>
        </div>
      </div>
      
      <div class="recommendations">
        <h4>📋 Recommendations:</h4>
        <ul>
          ${getRecommendations(confidence, isMisinformation).map(rec => `<li>${rec}</li>`).join('')}
        </ul>
      </div>
      
      <div class="action-buttons">
        <button class="btn btn-small btn-secondary" onclick="copyToClipboard('${escapeHtml(result.text || '')}')">
          📋 Copy Text
        </button>
        <button class="btn btn-small btn-secondary" onclick="shareResult()">
          🔗 Share
        </button>
        <button class="btn btn-small btn-secondary" onclick="reportFalsePositive()">
          ⚠️ Report
        </button>
      </div>
    </div>
  `;
  
  resultsDiv.classList.remove('hidden');
}

function getRecommendations(confidence, isMisinformation) {
  const recommendations = [];
  
  if (confidence > 0.85) {
    recommendations.push('🚩 Very high confidence in misinformation detection');
    recommendations.push('⛔ Avoid sharing this content');
    recommendations.push('📝 Look for fact-checks from reliable sources');
  } else if (confidence > 0.65) {
    recommendations.push('⚠️ Moderate risk detected');
    recommendations.push('🔍 Verify with additional sources');
    recommendations.push('❓ Check source credibility');
  } else {
    recommendations.push('✅ Appears to be factually sound');
    recommendations.push('💡 Still recommend verifying important claims');
    recommendations.push('🔗 Check source reliability for context');
  }
  
  return recommendations;
}

// ── Settings ────────────────────────────────────────────────
function loadSettings() {
  chrome.storage.sync.get(['apiUrl', 'autoCheck', 'confidenceThreshold'], (items) => {
    if (items.apiUrl) {
      apiUrl = items.apiUrl;
      apiUrlInput.value = apiUrl;
    }
    
    if (items.autoCheck) {
      autoCheckCheckbox.checked = items.autoCheck;
    }
    
    if (items.confidenceThreshold) {
      confidenceThresholdSelect.value = items.confidenceThreshold;
    }
  });
}

function saveSettings() {
  apiUrl = apiUrlInput.value.trim() || 'http://localhost:8000';
  
  const settings = {
    apiUrl: apiUrl,
    autoCheck: autoCheckCheckbox.checked,
    confidenceThreshold: confidenceThresholdSelect.value
  };
  
  chrome.storage.sync.set(settings, () => {
    settingsStatus.textContent = '✅ Settings saved!';
    settingsStatus.classList.remove('hidden');
    
    setTimeout(() => {
      settingsStatus.classList.add('hidden');
    }, 2000);
  });
}

// ── UI Helpers ──────────────────────────────────────────────
function showLoading(show) {
  if (show) {
    loadingDiv.classList.remove('hidden');
  } else {
    loadingDiv.classList.add('hidden');
  }
}

function hideResults() {
  resultsDiv.classList.add('hidden');
}

function showError(message) {
  errorDiv.textContent = message;
  errorDiv.classList.remove('hidden');
}

function hideError() {
  errorDiv.classList.add('hidden');
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    alert('Copied to clipboard!');
  });
}

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

function shareResult() {
  const text = textInput.value.substring(0, 100);
  const url = `https://afwaah-tracker.vercel.app/share?text=${encodeURIComponent(text)}`;
  
  if (navigator.share) {
    navigator.share({
      title: 'AFWAH Fact Check',
      text: 'I fact-checked this with AFWAH',
      url: url
    });
  } else {
    copyToClipboard(url);
  }
}

function reportFalsePositive() {
  // TODO: Implement reporting to backend
  alert('Thank you for your feedback! This helps improve our detection.');
}
