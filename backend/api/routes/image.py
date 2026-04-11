# ============================================================
# backend/api/routes/image.py
#
# POST /api/analyze-image  — AI Image Detection
# ───────────────────────────────────────────────
# Analyzes uploaded images to detect if they're AI-generated
# ============================================================

import logging
import os
import tempfile
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.ml.ai_image_detector import analyze_image_file

logger = logging.getLogger(__name__)
router = APIRouter()

# Upload size limit: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024


class ImageAnalysisResponse(BaseModel):
    status: str
    is_ai_generated: bool
    confidence: float
    markers: list
    details: str
    analyzed_at: datetime


@router.post(
    "/analyze-image",
    response_model=ImageAnalysisResponse,
    summary="Analyze image for AI generation",
    description="""
    Uploads and analyzes an image to detect if it's AI-generated.
    
    Returns:
    - **is_ai_generated**: Boolean indicating if image appears AI-generated
    - **confidence**: Confidence score (0-1)
    - **markers**: List of detected AI generation markers
    - **details**: Human-readable analysis explanation
    """,
)
async def analyze_image(file: UploadFile = File(...)) -> ImageAnalysisResponse:
    """
    Main image analysis endpoint.
    Accepts image upload and returns AI generation detection results.
    """
    
    # ── Validate file ──────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Accepted: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read file")
    
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"
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
        
        logger.info(f"Analyzing image: {file.filename}")
        
        # Analyze the image
        analysis_result = await analyze_image_file(temp_file_path)
        
        if analysis_result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Image analysis failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        return ImageAnalysisResponse(
            status="success",
            is_ai_generated=analysis_result.get("is_ai_generated", False),
            confidence=analysis_result.get("confidence", 0.0),
            markers=analysis_result.get("markers", []),
            details=analysis_result.get("details", "Analysis complete"),
            analyzed_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image analysis error: {str(e)}"
        )
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
