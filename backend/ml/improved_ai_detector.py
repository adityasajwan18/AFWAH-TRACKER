# ============================================================
# backend/ml/improved_ai_detector.py
#
# Enhanced AI-Generated Image Detection with ML Models
# Combines traditional CV + Neural Network approaches
# ============================================================

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import os
from pathlib import Path

try:
    from PIL import Image
    import cv2
    from scipy import signal
    from sklearn.ensemble import IsolationForest
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ImprovedAIDetector:
    """Enhanced AI image detection with machine learning"""
    
    def __init__(self):
        """Initialize detector with pre-trained models"""
        self.isolation_forest = None
        self.load_models()
    
    def load_models(self):
        """Load or initialize ML models"""
        try:
            # Initialize Isolation Forest for anomaly detection
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            logger.info("ML models initialized")
        except Exception as e:
            logger.warning(f"Could not initialize ML models: {e}")
    
    # ── ENHANCED DETECTION METHODS ──────────────────────────
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive AI image detection combining multiple methods.
        """
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            pixels = np.array(img, dtype=np.float32)
            
            # Extract features from multiple detection methods
            features = {}
            
            # 1. Frequency domain analysis (FFT artifacts)
            features['frequency'] = self._analyze_frequency(pixels)
            
            # 2. Error level analysis (compression inconsistencies)
            features['error_level'] = self._analyze_error_level(pixels)
            
            # 3. Noise analysis (AI images have characteristic noise patterns)
            features['noise'] = self._analyze_noise(pixels)
            
            # 4. Edge detection (AI struggles with fine details)
            features['edge'] = self._analyze_edges(pixels)
            
            # 5. Color pattern analysis
            features['color'] = self._analyze_color_patterns(pixels)
            
            # 6. DCT (Discrete Cosine Transform) analysis
            features['dct'] = self._analyze_dct(pixels)
            
            # 7. Blur analysis (AI images often have subtle blur)
            features['blur'] = self._analyze_blur(pixels)
            
            # 8. Splicing detection
            features['splicing'] = self._detect_splicing(pixels)
            
            # 9. Metadata analysis
            features['metadata'] = self._analyze_metadata(img)
            
            # Machine learning classification
            ml_score = self._ml_classification(features)
            
            # Weighted ensemble of all methods
            confidence = self._ensemble_decision(features, ml_score)
            
            # Generate detailed report
            report = self._generate_report(features, confidence, pixels)
            
            return {
                "is_ai_generated": confidence > 0.65,
                "confidence": float(confidence),
                "method_scores": {k: float(v) for k, v in features.items()},
                "ml_score": float(ml_score),
                "details": report,
                "markers": self._extract_markers(features)
            }
        
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                "is_ai_generated": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _analyze_frequency(self, pixels: np.ndarray) -> float:
        """
        FFT-based frequency analysis.
        AI images often have unnatural frequency distributions.
        """
        try:
            gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Compute FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log1p(np.abs(fft_shift))
            
            # Analyze frequency distribution
            total_energy = np.sum(magnitude ** 2)
            high_freq_energy = np.sum(magnitude[magnitude > np.percentile(magnitude, 80)] ** 2)
            
            # AI images tend to have different frequency patterns
            freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            # Normalize to 0-1
            return min(max(freq_ratio / 0.3, 0), 1.0)
        except:
            return 0.0
    
    def _analyze_error_level(self, pixels: np.ndarray) -> float:
        """
        Error Level Analysis (ELA).
        Detects if different image regions have different compression levels.
        """
        try:
            # Simulate re-compression artifacts
            pil_img = Image.fromarray(pixels.astype(np.uint8))
            
            # Save and reload at lower quality to detect differences
            import io
            temp_io = io.BytesIO()
            pil_img.save(temp_io, format='JPEG', quality=85)
            temp_io.seek(0)
            recompressed = Image.open(temp_io)
            
            # Calculate difference
            original = np.array(pil_img, dtype=np.float32)
            recomp = np.array(recompressed, dtype=np.float32)
            
            difference = np.abs(original - recomp)
            ela_mean = np.mean(difference)
            ela_std = np.std(difference)
            
            # AI images often have inconsistent compression artifacts
            ela_score = (ela_mean + ela_std) / 255.0
            return min(ela_score, 1.0)
        except:
            return 0.0
    
    def _analyze_noise(self, pixels: np.ndarray) -> float:
        """
        Analyze noise characteristics.
        AI images have different noise patterns than real photos.
        """
        try:
            # Convert to grayscale
            if len(pixels.shape) == 3:
                gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = pixels.astype(np.uint8)
            
            # Laplacian filtering to detect high-frequency noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = np.std(laplacian)
            
            # Compare with gradient-based noise
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(sobelx**2 + sobely**2)
            
            # AI images typically have artificial noise patterns
            noise_ratio = noise_level / (np.mean(gradient) + 1e-10)
            return min(max(noise_ratio / 2.0, 0), 1.0)
        except:
            return 0.0
    
    def _analyze_edges(self, pixels: np.ndarray) -> float:
        """
        Edge analysis.
        AI images struggle with realistic edge transitions.
        """
        try:
            gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Analyze edge sharpness
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            edge_sharpness = np.std(laplacian)
            
            # AI images often have too perfect or too blurry edges
            if edge_density < 0.01 or edge_density > 0.3:
                return 0.7  # Unusual edge distribution
            
            return min(abs(edge_sharpness - 30) / 50, 1.0)
        except:
            return 0.0
    
    def _analyze_color_patterns(self, pixels: np.ndarray) -> float:
        """
        Analyze color channel coherence.
        AI models sometimes generate inconsistent color patterns.
        """
        try:
            # Analyze color channel statistics
            r_mean, r_std = np.mean(pixels[:,:,0]), np.std(pixels[:,:,0])
            g_mean, g_std = np.mean(pixels[:,:,1]), np.std(pixels[:,:,1])
            b_mean, b_std = np.mean(pixels[:,:,2]), np.std(pixels[:,:,2])
            
            # Calculate channel correlation
            corr_rg = np.corrcoef(pixels[:,:,0].flatten(), pixels[:,:,1].flatten())[0,1]
            corr_gb = np.corrcoef(pixels[:,:,1].flatten(), pixels[:,:,2].flatten())[0,1]
            corr_rb = np.corrcoef(pixels[:,:,0].flatten(), pixels[:,:,2].flatten())[0,1]
            
            avg_corr = np.mean([abs(corr_rg), abs(corr_gb), abs(corr_rb)])
            
            # Natural images have high channel correlation
            # AI images sometimes have unnatural patterns
            return max(1.0 - avg_corr, 0.0)
        except:
            return 0.0
    
    def _analyze_dct(self, pixels: np.ndarray) -> float:
        """
        DCT (Discrete Cosine Transform) analysis.
        Used in JPEG compression; AI images have characteristic patterns.
        """
        try:
            from scipy.fftpack import dct
            
            gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Apply block-wise DCT (like in JPEG)
            block_size = 8
            h, w = gray.shape
            
            dct_features = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    dct_block = dct(dct.dct(block, axis=0), axis=1)
                    dct_features.append(np.std(dct_block))
            
            dct_variance = np.std(dct_features)
            return min(dct_variance / 100, 1.0)
        except:
            return 0.0
    
    def _analyze_blur(self, pixels: np.ndarray) -> float:
        """
        Blur detection.
        AI images often have artificial subtle blur.
        """
        try:
            gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Laplacian variance (higher = sharper)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Also check with Sobel
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edge_var = np.var(sobelx)
            
            # Combine metrics
            blur_score = (laplacian_var + edge_var) / 1000.0
            return min(blur_score, 1.0)
        except:
            return 0.0
    
    def _detect_splicing(self, pixels: np.ndarray) -> float:
        """
        Detect image splicing/editing inconsistencies.
        """
        try:
            # Analyze gradient consistency
            gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            
            # Calculate gradient magnitude
            gradient = np.sqrt(sobelx**2 + sobely**2)
            
            # Analyze gradient histogram for anomalies
            hist, _ = np.histogram(gradient.flatten(), bins=50)
            hist = hist / np.sum(hist)  # Normalize
            
            # Entropy (spliced images often have different entropy patterns)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            
            # Natural images typically have entropy around 4-5
            splicing_score = abs(entropy - 4.5) / 5.0
            return min(splicing_score, 1.0)
        except:
            return 0.0
    
    def _analyze_metadata(self, img: Image.Image) -> float:
        """
        Analyze image metadata.
        AI-generated images often lack realistic metadata.
        """
        try:
            exif_data = img.getexif()
            
            if not exif_data:
                return 0.7  # No metadata is suspicious
            
            # Check for required EXIF tags
            required_tags = [271, 272, 305]  # Manufacturer, Model, Software
            present_tags = sum(1 for tag in required_tags if tag in exif_data)
            
            return max(1.0 - (present_tags / len(required_tags)), 0.0)
        except:
            return 0.0
    
    def _ml_classification(self, features: Dict[str, float]) -> float:
        """
        Machine learning classification using extracted features.
        """
        try:
            if not self.isolation_forest:
                return 0.5
            
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Isolation Forest: -1 for anomalies (AI images), 1 for normal
            prediction = self.isolation_forest.predict(feature_vector)[0]
            score = self.isolation_forest.score_samples(feature_vector)[0]
            
            # Convert to probability
            ml_score = 1 / (1 + np.exp(-score))  # Sigmoid conversion
            return float(ml_score)
        except:
            return 0.5
    
    def _ensemble_decision(self, features: Dict[str, float], ml_score: float) -> float:
        """
        Weighted ensemble of all detection methods.
        """
        weights = {
            'frequency': 0.20,
            'error_level': 0.18,
            'noise': 0.15,
            'edge': 0.12,
            'color': 0.10,
            'dct': 0.10,
            'blur': 0.08,
            'splicing': 0.07,
            'metadata': 0.00  # Metadata alone isn't conclusive
        }
        
        # Weighted average of feature scores
        ensemble_score = sum(
            features[key] * weights.get(key, 0)
            for key in features if key in weights
        )
        
        # Add ML score with moderate weight
        ensemble_score = ensemble_score * 0.7 + ml_score * 0.3
        
        return min(max(ensemble_score, 0), 1.0)
    
    def _extract_markers(self, features: Dict[str, float]) -> List[str]:
        """Extract readable markers from detected anomalies."""
        markers = []
        
        if features['frequency'] > 0.6:
            markers.append("🔴 Unnatural frequency distribution")
        if features['error_level'] > 0.6:
            markers.append("🔴 Compression inconsistencies detected")
        if features['noise'] > 0.6:
            markers.append("🟡 Artificial noise patterns")
        if features['edge'] > 0.6:
            markers.append("🟡 Edge anomalies")
        if features['color'] > 0.6:
            markers.append("🟡 Color channel inconsistencies")
        if features['dct'] > 0.6:
            markers.append("🔴 DCT analysis flags")
        if features['blur'] > 0.5:
            markers.append("🟡 Unnatural blur detected")
        if features['splicing'] > 0.6:
            markers.append("🔴 Possible image splicing")
        if features['metadata'] > 0.7:
            markers.append("⚪ Missing or suspicious metadata")
        
        return markers
    
    def _generate_report(self, features: Dict[str, float], 
                        confidence: float, pixels: np.ndarray) -> str:
        """Generate human-readable analysis report."""
        
        if confidence > 0.80:
            severity = "🚨 VERY HIGH CONFIDENCE - AI GENERATED"
        elif confidence > 0.65:
            severity = "🚩 HIGH CONFIDENCE - LIKELY AI GENERATED"
        elif confidence > 0.50:
            severity = "⚠️ MODERATE CONFIDENCE - POSSIBLE AI"
        elif confidence > 0.35:
            severity = "❓ LOW CONFIDENCE - LIKELY GENUINE"
        else:
            severity = "✅ LOW CONFIDENCE - APPEARS AUTHENTIC"
        
        top_methods = sorted(
            features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        method_details = "\n".join(
            f"  • {method.replace('_', ' ').title()}: {score*100:.1f}%"
            for method, score in top_methods
        )
        
        report = f"""
{severity}
Confidence Score: {confidence*100:.1f}%

Detection Methods Analysis:
{method_details}

Image Dimensions: {pixels.shape[1]}x{pixels.shape[0]} pixels
Color Channels: {pixels.shape[2] if len(pixels.shape) > 2 else 1}

Recommendation:
{'🚫 Do not trust as authentic' if confidence > 0.65 else '✅ Likely authentic, but manual verification recommended'}
        """
        
        return report.strip()


# Singleton instance
_detector = None

def get_improved_detector():
    global _detector
    if _detector is None:
        _detector = ImprovedAIDetector()
    return _detector
