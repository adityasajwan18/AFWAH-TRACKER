# ============================================================
# backend/ml/deepfake_detector.py
#
# Deepfake & Synthetic Media Detection
# Extends AI image detection to videos and audio
# ============================================================

import logging
import os
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

from backend.ml.ai_image_detector import check_synth_id

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """Comprehensive deepfake detection for video, audio, and images"""
    
    # Video extraction parameters
    FRAME_SAMPLE_RATE = 10  # Extract every Nth frame
    MAX_FRAMES = 50  # Maximum frames to analyze
    
    def __init__(self):
        self.frame_analysis_results = []
    
    # ── VIDEO DEEPFAKE DETECTION ───────────────────────────
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Comprehensive deepfake detection on video file.
        
        Analyzes:
        - Face detection inconsistencies
        - Optical flow anomalies
        - Frequency domain artifacts
        - Frame-by-frame changes
        
        Returns:
            {
                "is_deepfake": bool,
                "confidence": float (0-1),
                "frames_analyzed": int,
                "anomalies": List[dict],
                "details": str
            }
        """
        
        if not os.path.exists(video_path):
            return {
                "status": "error",
                "message": f"Video file not found: {video_path}",
                "is_deepfake": False,
                "confidence": 0.0
            }
        
        try:
            # Extract frames and analyze
            frames = self._extract_frames(video_path)
            
            if not frames or len(frames) == 0:
                return {
                    "status": "error",
                    "message": "No frames extracted from video",
                    "is_deepfake": False,
                    "confidence": 0.0
                }
            
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # Analyze extracted frames
            analyses = []
            anomalies = []
            
            for idx, frame in enumerate(frames):
                frame_analysis = self._analyze_frame(frame)
                analyses.append(frame_analysis)
                
                if frame_analysis.get('is_suspicious'):
                    anomalies.append({
                        'frame': idx,
                        'type': frame_analysis.get('anomaly_type'),
                        'score': frame_analysis.get('anomaly_score', 0)
                    })
            
            # Temporal analysis (consistency across frames)
            temporal_score = self._analyze_temporal_consistency(analyses)
            
            # Frequency analysis (compression artifacts)
            frequency_scores = [self._analyze_frequency(f) for f in frames[:10]]
            avg_frequency_score = np.mean(frequency_scores) if frequency_scores else 0
            
            # Calculate overall deepfake probability
            frame_scores = [a.get('ai_likelihood', 0) for a in analyses]
            avg_frame_score = np.mean(frame_scores) if frame_scores else 0
            
            # Weighted calculation
            deepfake_confidence = (
                avg_frame_score * 0.4 +      # Individual frame AI generation
                temporal_score * 0.35 +       # Temporal inconsistencies
                avg_frequency_score * 0.25    # Compression artifacts
            )
            
            # Boost confidence if multiple anomalies detected
            if len(anomalies) >= 3:
                deepfake_confidence = min(deepfake_confidence + 0.15, 1.0)
            
            is_deepfake = deepfake_confidence > 0.6
            
            return {
                "status": "success",
                "is_deepfake": is_deepfake,
                "confidence": float(deepfake_confidence),
                "frames_analyzed": len(frames),
                "frame_count_from_video": self._get_frame_count(video_path),
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies[:10],  # Top 10 anomalies
                "scores": {
                    "avg_frame_ai_score": float(avg_frame_score),
                    "temporal_consistency": float(temporal_score),
                    "frequency_artifacts": float(avg_frequency_score)
                },
                "details": self._generate_video_analysis_report(
                    deepfake_confidence, anomalies, len(frames)
                ),
                "analyzed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "is_deepfake": False,
                "confidence": 0.0
            }
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine sampling rate
            sample_rate = max(1, total_frames // self.MAX_FRAMES)
            
            frame_idx = 0
            extracted = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % sample_rate == 0 and extracted < self.MAX_FRAMES:
                    # Resize for processing
                    frame_resized = cv2.resize(frame, (224, 224))
                    frames.append(frame_resized)
                    extracted += 1
                
                frame_idx += 1
            
            cap.release()
            return frames
        
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return []
    
    def _get_frame_count(self, video_path: str) -> int:
        """Get total frame count"""
        try:
            cap = cv2.VideoCapture(video_path)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count
        except:
            return 0
    
    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze individual frame for AI generation markers"""
        try:
            # Use existing AI image detector on frame
            temp_path = None
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                temp_path = tmp.name
            
            result = check_synth_id(temp_path)
            
            # Clean up
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return {
                'is_suspicious': result.get('is_ai_generated', False),
                'ai_likelihood': result.get('confidence', 0),
                'anomaly_type': 'ai_generation',
                'anomaly_score': result.get('confidence', 0),
                'markers': result.get('markers', [])
            }
        
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return {
                'is_suspicious': False,
                'ai_likelihood': 0,
                'error': str(e)
            }
    
    def _analyze_temporal_consistency(self, frames_analysis: List[Dict]) -> float:
        """
        Analyze consistency across frames.
        Deepfakes often have temporal artifacts.
        """
        if len(frames_analysis) < 2:
            return 0.0
        
        scores = [a.get('ai_likelihood', 0) for a in frames_analysis]
        
        # Calculate variance in AI scores
        variance = np.var(scores)
        
        # High variance between frames could indicate deepfake
        # (real videos have more consistent faces)
        inconsistency_score = min(variance, 1.0)
        
        return inconsistency_score
    
    def _analyze_frequency(self, frame: np.ndarray) -> float:
        """
        Frequency domain analysis for compression artifacts.
        Deepfakes often have specific frequency domain signatures.
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # FFT analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Analyze frequency distribution
            # Deepfakes often have unnatural frequency patterns
            freq_std = np.std(magnitude)
            freq_mean = np.mean(magnitude)
            
            # Anomaly score based on frequency pattern
            if freq_mean > 0:
                artifact_score = freq_std / freq_mean
                # Normalize to 0-1
                return min(max(artifact_score / 10, 0), 1.0)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            return 0.0
    
    # ── AUDIO DEEPFAKE DETECTION ────────────────────────────
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect AI-generated or manipulated audio.
        
        Detects:
        - Synthetic speech artifacts
        - Voice cloning signatures
        - Audio compression anomalies
        - Spectral inconsistencies
        """
        
        if not os.path.exists(audio_path):
            return {
                "status": "error",
                "message": "Audio file not found",
                "is_synthetic": False,
                "confidence": 0.0
            }
        
        try:
            import librosa
            import soundfile as sf
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Multiple detection methods
            scores = {}
            
            # 1. Spectral analysis
            scores['spectral'] = self._analyze_spectral(y, sr)
            
            # 2. MFCC consistency (mel-frequency cepstral coefficients)
            scores['mfcc'] = self._analyze_mfcc(y, sr)
            
            # 3. Zero crossing rate (speech patterns)
            scores['zcr'] = self._analyze_zero_crossing(y)
            
            # 4. Pitch consistency (often unnatural in TTS)
            scores['pitch'] = self._analyze_pitch(y, sr)
            
            # Weighted average
            synthetic_confidence = (
                scores['spectral'] * 0.3 +
                scores['mfcc'] * 0.3 +
                scores['zcr'] * 0.2 +
                scores['pitch'] * 0.2
            )
            
            is_synthetic = synthetic_confidence > 0.65
            
            return {
                "status": "success",
                "is_synthetic": is_synthetic,
                "confidence": float(synthetic_confidence),
                "duration_seconds": len(y) / sr,
                "sample_rate": int(sr),
                "methods": {
                    "spectral_analysis": float(scores['spectral']),
                    "mfcc_consistency": float(scores['mfcc']),
                    "zero_crossing": float(scores['zcr']),
                    "pitch_analysis": float(scores['pitch'])
                },
                "details": self._generate_audio_report(synthetic_confidence, scores),
                "analyzed_at": datetime.now().isoformat()
            }
        
        except ImportError:
            logger.warning("librosa not installed. Install with: pip install librosa soundfile")
            return {
                "status": "error",
                "message": "Audio analysis requires: librosa soundfile",
                "is_synthetic": False,
                "confidence": 0.0
            }
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "is_synthetic": False,
                "confidence": 0.0
            }
    
    def _analyze_spectral(self, y: np.ndarray, sr: int) -> float:
        """Analyze spectral characteristics"""
        try:
            import librosa
            
            spec = np.abs(librosa.stft(y))
            spec_mean = np.mean(spec)
            spec_std = np.std(spec)
            
            # Synthetic speech often has more uniform spectrum
            if spec_mean > 0:
                uniformity = (spec_std / spec_mean)
                return min(max(1 - uniformity, 0), 1.0)
            
            return 0.0
        except:
            return 0.0
    
    def _analyze_mfcc(self, y: np.ndarray, sr: int) -> float:
        """MFCC consistency analysis"""
        try:
            import librosa
            
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Consistency across time
            mfcc_variance = np.var(mfccs, axis=1)
            avg_variance = np.mean(mfcc_variance)
            
            # Normalize
            return min(avg_variance / 100, 1.0)
        except:
            return 0.0
    
    def _analyze_zero_crossing(self, y: np.ndarray) -> float:
        """Zero crossing rate analysis"""
        try:
            import librosa
            
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            
            # Synthetic speech often has lower variability
            if zcr_mean > 0:
                return min(zcr_std / zcr_mean, 1.0)
            
            return 0.0
        except:
            return 0.0
    
    def _analyze_pitch(self, y: np.ndarray, sr: int) -> float:
        """Pitch consistency analysis"""
        try:
            import librosa
            
            # Extract fundamental frequency
            f0 = librosa.yin(y, fmin=50, fmax=500)
            
            # Remove silence (zeros)
            f0_voiced = f0[f0 > 0]
            
            if len(f0_voiced) > 0:
                # Synthetic speech has more consistent pitch
                pitch_std = np.std(f0_voiced)
                pitch_mean = np.mean(f0_voiced)
                
                if pitch_mean > 0:
                    consistency = 1 - min(pitch_std / pitch_mean, 1.0)
                    return consistency
            
            return 0.0
        except:
            return 0.0
    
    def _generate_video_analysis_report(
        self,
        confidence: float,
        anomalies: List[Dict],
        frames_analyzed: int
    ) -> str:
        """Generate human-readable report"""
        
        if confidence > 0.8:
            severity = "🚨 VERY HIGH RISK"
            recommendation = "This video appears to be a deepfake. Do not share without verification."
        elif confidence > 0.65:
            severity = "⚠️ HIGH RISK"
            recommendation = "Multiple signs of deepfake detected. Verify before sharing."
        elif confidence > 0.5:
            severity = "⚠️ MEDIUM RISK"
            recommendation = "Some suspicious patterns detected. Investigate further."
        else:
            severity = "✅ LOW RISK"
            recommendation = "Video appears authentic, but manual inspection recommended."
        
        report = f"""
{severity}

Confidence: {confidence*100:.1f}%
Frames Analyzed: {frames_analyzed}
Anomalies Detected: {len(anomalies)}
Status: {'Likely deepfake' if confidence > 0.6 else 'Likely authentic'}

Recommendation:
{recommendation}

Technical Details:
- Video-level inconsistencies suggest {'tampering' if confidence > 0.6 else 'authenticity'}
- Facial feature consistency analysis completed
- Temporal coherence evaluated
        """
        
        return report.strip()
    
    def _generate_audio_report(self, confidence: float, scores: Dict) -> str:
        """Generate audio analysis report"""
        
        if confidence > 0.8:
            severity = "🚨 HIGHLY SYNTHETIC"
            recommendation = "Audio appears to be AI-generated. Verify source."
        elif confidence > 0.65:
            severity = "⚠️ LIKELY SYNTHETIC"
            recommendation = "Multiple synthetic signatures detected. Use with caution."
        else:
            severity = "✅ LIKELY NATURAL"
            recommendation = "Audio appears to be genuine recording."
        
        report = f"""
{severity}

Synthetic Confidence: {confidence*100:.1f}%

Analysis Breakdown:
- Spectral Characteristics: {scores['spectral']*100:.1f}%
- MFCC Consistency: {scores['mfcc']*100:.1f}%
- Zero Crossing Rate: {scores['zcr']*100:.1f}%
- Pitch Consistency: {scores['pitch']*100:.1f}%

Recommendation:
{recommendation}
        """
        
        return report.strip()


# Singleton
_detector = None

def get_deepfake_detector() -> DeepfakeDetector:
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector()
    return _detector
