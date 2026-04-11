# ============================================================
# backend/api/routes/media.py
#
# POST /api/analyze-video    — Deepfake video detection
# POST /api/analyze-audio    — Synthetic audio detection
# ───────────────────────────────────────────────────────────
# ============================================================

import logging
import os
import tempfile
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.ml.deepfake_detector import get_deepfake_detector

logger = logging.getLogger(__name__)
router = APIRouter()

# Upload size limits
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50MB


class VideoAnalysisResponse(BaseModel):
    status: str
    is_deepfake: bool
    confidence: float
    frames_analyzed: int
    anomalies_detected: int
    anomalies: list
    scores: dict
    details: str
    analyzed_at: datetime


class AudioAnalysisResponse(BaseModel):
    status: str
    is_synthetic: bool
    confidence: float
    duration_seconds: float
    methods: dict
    details: str
    analyzed_at: datetime


# ── VIDEO ANALYSIS ─────────────────────────────────────────

@router.post(
    "/analyze-video",
    response_model=dict,
    summary="Analyze video for deepfakes",
    description="""
    Upload and analyze a video file for deepfake signatures.
    
    Detects:
    - Face generation artifacts
    - Temporal inconsistencies
    - Compression anomalies
    - Optical flow anomalies
    
    Returns:
    - **is_deepfake**: Boolean indicating if video is a deepfake
    - **confidence**: Confidence score (0-1)
    - **frames_analyzed**: Number of frames analyzed
    - **anomalies**: List of detected anomalies
    - **details**: Human-readable analysis report
    """,
)
async def analyze_video(file: UploadFile = File(...)) -> dict:
    """
    Main video analysis endpoint.
    Accepts video upload and returns deepfake detection results.
    """
    
    # ── Validate file ──────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Video format not supported. Accepted: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")
    
    # Check file size
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_VIDEO_SIZE // 1024 // 1024}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # ── Save temp file and analyze ────────────────────────────
    temp_file_path = None
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            temp_file_path = tmp.name
        
        logger.info(f"Analyzing video: {file.filename}")
        
        # Analyze the video
        detector = get_deepfake_detector()
        analysis_result = detector.analyze_video(temp_file_path)
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


# ── AUDIO ANALYSIS ─────────────────────────────────────────

@router.post(
    "/analyze-audio",
    response_model=dict,
    summary="Analyze audio for synthetic/deepfake speech",
    description="""
    Upload and analyze audio file for AI-generated speech detection.
    
    Detects:
    - Synthetic speech signatures
    - Voice cloning artifacts
    - Text-to-speech patterns
    - Unnatural pitch/tone patterns
    
    Returns:
    - **is_synthetic**: Boolean indicating if audio is AI-generated
    - **confidence**: Confidence score (0-1)
    - **methods**: Individual detection method scores
    - **details**: Human-readable analysis report
    """,
)
async def analyze_audio(file: UploadFile = File(...)) -> dict:
    """
    Main audio analysis endpoint.
    Accepts audio upload and returns synthetic speech detection results.
    """
    
    # ── Validate file ──────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    allowed_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Audio format not supported. Accepted: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")
    
    # Check file size
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_AUDIO_SIZE // 1024 // 1024}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # ── Save temp file and analyze ────────────────────────────
    temp_file_path = None
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            temp_file_path = tmp.name
        
        logger.info(f"Analyzing audio: {file.filename}")
        
        # Analyze the audio
        detector = get_deepfake_detector()
        analysis_result = detector.analyze_audio(temp_file_path)
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio analysis failed: {str(e)}"
        )
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


# ── COMBINED MEDIA ANALYSIS ────────────────────────────────

@router.post(
    "/analyze-media",
    summary="Analyze video + audio for deepfakes",
    description="""
    Upload a video file and analyze both video AND audio streams.
    Combines deepfake video detection with synthetic audio detection
    for comprehensive multimedia verification.
    
    Returns combined analysis of visual and audio components.
    """,
)
async def analyze_media(file: UploadFile = File(...)) -> dict:
    """
    Comprehensive media analysis combining video and audio detection.
    """
    
    # ── Validate file ──────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Media format not supported. Accepted: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read file")
    
    # Check file size
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_VIDEO_SIZE // 1024 // 1024}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    # ── Save temp file and analyze ────────────────────────────
    temp_file_path = None
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            temp_file_path = tmp.name
        
        logger.info(f"Analyzing media: {file.filename}")
        
        detector = get_deepfake_detector()
        
        # Analyze video
        video_result = detector.analyze_video(temp_file_path)
        
        # Analyze audio
        audio_result = detector.analyze_audio(temp_file_path)
        
        # Combine results
        overall_confidence = max(
            video_result.get('confidence', 0),
            audio_result.get('confidence', 0)
        )
        
        return {
            "status": "success",
            "is_deepfake": overall_confidence > 0.6,
            "confidence": overall_confidence,
            "video_analysis": video_result,
            "audio_analysis": audio_result,
            "combined_verdict": (
                "🚨 HIGH RISK - Multiple deepfake signatures detected" if overall_confidence > 0.75
                else "⚠️ MEDIUM RISK - Some suspicious patterns detected" if overall_confidence > 0.55
                else "✅ LOW RISK - Appears authentic"
            ),
            "recommendation": (
                "Do not share without verification" if overall_confidence > 0.75
                else "Verify with additional sources" if overall_confidence > 0.55
                else "Likely authentic media"
            ),
            "analyzed_at": datetime.now(timezone.utc).isoformat()
        }
    
    except Exception as e:
        logger.error(f"Media analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Media analysis failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


# ── URL-based Analysis (optional) ──────────────────────────

@router.post(
    "/analyze-media-url",
    summary="Analyze media from URL",
    description="Download and analyze media from a URL",
)
async def analyze_media_url(url: str, media_type: str = "video"):
    """
    Download media from URL and analyze it.
    Useful for analyzing social media videos without explicit upload.
    
    media_type: 'video' or 'audio'
    """
    
    try:
        import aiohttp
        
        # Download file
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to download file")
                
                content = await resp.read()
                
                if len(content) == 0:
                    raise HTTPException(status_code=400, detail="Downloaded file is empty")
        
        # Determine file extension from URL
        path = url.split('?')[0]
        file_ext = os.path.splitext(path)[1] or '.mp4'
        
        temp_file_path = None
        
        try:
            # Save and analyze
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(content)
                temp_file_path = tmp.name
            
            detector = get_deepfake_detector()
            
            if media_type.lower() == 'audio':
                result = detector.analyze_audio(temp_file_path)
            else:
                result = detector.analyze_video(temp_file_path)
            
            return result
        
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    except Exception as e:
        logger.error(f"URL analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
