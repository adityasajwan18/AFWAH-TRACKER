# ============================================================
# backend/api/routes/social_media.py
#
# Social Media Integration Endpoints
# /api/social/* — Real-time monitoring & analysis
# ============================================================

import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.integrations.realtime_monitor import get_monitor
from backend.ml.classifier import classify_text
from backend.ml.rumor_analyzer import analyze_rumor

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request/Response Models ────────────────────────────────

class StartMonitoringRequest(BaseModel):
    platform: str  # 'twitter' or 'reddit'
    query: str
    poll_interval: int = 60


class MonitoringResponse(BaseModel):
    task_id: str
    platform: str
    query: str
    status: str
    started_at: datetime
    posts_analyzed: int


class SocialPostResponse(BaseModel):
    post_id: str
    platform: str
    author: str
    content: str
    timestamp: datetime
    url: str
    engagement: dict


class MonitoredPostAnalysis(BaseModel):
    post_id: str
    content: str
    is_misinformation: bool
    confidence: float
    classification: str
    platform: str
    author: str
    url: str
    engagement: dict


# ── Configuration Endpoints ────────────────────────────────

@router.post("/social/configure/twitter")
async def configure_twitter(bearer_token: str):
    """
    Configure Twitter API credentials.
    Get token from: https://developer.twitter.com/en/portal/dashboard
    """
    try:
        monitor = get_monitor()
        monitor.register_twitter(bearer_token)
        return {
            "status": "✅ Twitter configured",
            "platform": "twitter"
        }
    except Exception as e:
        logger.error(f"Twitter config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social/configure/reddit")
async def configure_reddit(
    client_id: str,
    client_secret: str,
    user_agent: str = "AfwaahTracker/1.0"
):
    """
    Configure Reddit API credentials.
    Get from: https://www.reddit.com/prefs/apps
    """
    try:
        monitor = get_monitor()
        monitor.register_reddit(client_id, client_secret, user_agent)
        return {
            "status": "✅ Reddit configured",
            "platform": "reddit"
        }
    except Exception as e:
        logger.error(f"Reddit config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Monitoring Control ────────────────────────────────────

@router.post("/social/monitor/start", response_model=dict)
async def start_monitoring(
    request: StartMonitoringRequest,
    background_tasks: BackgroundTasks
):
    """
    Start real-time monitoring of social media platform.
    
    Args:
        platform: 'twitter' or 'reddit'
        query: Search query/keywords to monitor
        poll_interval: Seconds between polls (default: 60)
    
    Returns:
        task_id: ID of created monitoring task
    """
    try:
        monitor = get_monitor()
        
        if request.platform not in ['twitter', 'reddit']:
            raise ValueError("Platform must be 'twitter' or 'reddit'")
        
        task_id = await monitor.start_monitoring(
            platform=request.platform,
            query=request.query,
            poll_interval=request.poll_interval
        )
        
        logger.info(f"Started monitoring: {task_id}")
        
        return {
            "status": "✅ Monitoring started",
            "task_id": task_id,
            "platform": request.platform,
            "query": request.query
        }
    
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social/monitor/pause/{task_id}")
async def pause_monitoring(task_id: str):
    """Pause an active monitoring task"""
    try:
        monitor = get_monitor()
        monitor.pause_monitoring(task_id)
        
        return {"status": "✅ Monitoring paused", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social/monitor/resume/{task_id}")
async def resume_monitoring(task_id: str):
    """Resume a paused monitoring task"""
    try:
        monitor = get_monitor()
        monitor.resume_monitoring(task_id)
        
        return {"status": "✅ Monitoring resumed", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social/monitor/stop/{task_id}")
async def stop_monitoring(task_id: str):
    """Stop and remove a monitoring task"""
    try:
        monitor = get_monitor()
        monitor.stop_monitoring(task_id)
        
        return {"status": "✅ Monitoring stopped", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Monitoring Status ──────────────────────────────────────

@router.get("/social/monitor/status/{task_id}")
async def get_monitoring_status(task_id: str) -> dict:
    """Get current status of a monitoring task"""
    try:
        monitor = get_monitor()
        status = monitor.get_monitoring_status(task_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/social/monitor/tasks")
async def list_monitoring_tasks() -> List[dict]:
    """List all active monitoring tasks"""
    try:
        monitor = get_monitor()
        return monitor.list_monitoring_tasks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Monitored Posts ────────────────────────────────────────

@router.get("/social/posts")
async def get_monitored_posts(
    platform: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 100
) -> List[dict]:
    """
    Retrieve monitored posts from database.
    
    Args:
        platform: Filter by 'twitter' or 'reddit'
        query: Filter by search query
        limit: Maximum results (default: 100)
    """
    try:
        monitor = get_monitor()
        posts = monitor.get_monitored_posts(
            platform=platform,
            query=query,
            limit=limit
        )
        
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search (One-time) ──────────────────────────────────────

@router.get("/social/search/{platform}")
async def search_social_media(
    platform: str,
    query: str,
    limit: int = 100
):
    """
    One-time search of social media platform.
    Does not start continuous monitoring.
    
    Args:
        platform: 'twitter' or 'reddit'
        query: Search query
        limit: Maximum results
    """
    try:
        monitor = get_monitor()
        
        if platform not in monitor.clients:
            raise ValueError(f"Platform '{platform}' not configured")
        
        client = monitor.clients[platform]
        posts = await client.search_posts(query, limit=limit)
        
        return {
            "platform": platform,
            "query": query,
            "results_count": len(posts),
            "posts": [
                {
                    "post_id": p.post_id,
                    "author": p.author,
                    "content": p.content[:200],
                    "timestamp": p.timestamp.isoformat(),
                    "url": p.url,
                    "engagement": p.engagement
                }
                for p in posts
            ]
        }
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/social/trending/{platform}")
async def get_trending(platform: str, limit: int = 50):
    """
    Get trending posts from a social media platform.
    
    Args:
        platform: 'twitter' or 'reddit'
        limit: Maximum results
    """
    try:
        monitor = get_monitor()
        
        if platform not in monitor.clients:
            raise ValueError(f"Platform '{platform}' not configured")
        
        client = monitor.clients[platform]
        posts = await client.get_trending(limit=limit)
        
        return {
            "platform": platform,
            "trending_count": len(posts),
            "posts": [
                {
                    "post_id": p.post_id,
                    "author": p.author,
                    "content": p.content[:200],
                    "timestamp": p.timestamp.isoformat(),
                    "url": p.url,
                    "engagement": p.engagement
                }
                for p in posts
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Quick Analysis ────────────────────────────────────────

@router.post("/social/analyze")
async def analyze_social_post(
    platform: str,
    post_id: str,
    content: str,
    author: str = None,
):
    """
    Quick analysis of a social media post for misinformation.
    """
    try:
        # Use existing classifiers
        classification = await classify_text(content)
        rumor_analysis = await analyze_rumor(content)
        
        return {
            "post_id": post_id,
            "platform": platform,
            "author": author,
            "content": content[:200],
            "classification": classification,
            "rumor_analysis": rumor_analysis,
            "analyzed_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
