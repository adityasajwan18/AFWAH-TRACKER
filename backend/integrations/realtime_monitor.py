# ============================================================
# backend/integrations/realtime_monitor.py
#
# Real-time Monitoring Service
# Tracks social media posts and triggers analysis
# ============================================================

import logging
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime
import sqlite3
from dataclasses import dataclass

from backend.integrations.social_media_client import (
    SocialMediaClient, TwitterClient, RedditClient, SocialPost
)

logger = logging.getLogger(__name__)


@dataclass
class MonitoringTask:
    """Represents an active monitoring task"""
    task_id: str
    platform: str  # 'twitter', 'reddit'
    query: str
    status: str  # 'running', 'paused', 'stopped'
    started_at: datetime
    posts_analyzed: int = 0


class RealtimeMonitor:
    """Manages real-time social media monitoring"""
    
    def __init__(self, db_path: str = "./monitoring_tasks.db"):
        self.db_path = db_path
        self.tasks: Dict[str, MonitoringTask] = {}
        self.clients: Dict[str, SocialMediaClient] = {}
        self.analysis_callbacks: List[Callable] = []
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database for monitoring history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitored_posts (
                id TEXT PRIMARY KEY,
                platform TEXT,
                query TEXT,
                author TEXT,
                content TEXT,
                timestamp DATETIME,
                url TEXT,
                engagement TEXT,
                analysis_result TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitoring_tasks (
                task_id TEXT PRIMARY KEY,
                platform TEXT,
                query TEXT,
                status TEXT,
                started_at DATETIME,
                stopped_at DATETIME,
                posts_analyzed INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_twitter(self, bearer_token: str):
        """Register Twitter client"""
        self.clients['twitter'] = TwitterClient(bearer_token)
        logger.info("Twitter client registered")
    
    def register_reddit(self, client_id: str, client_secret: str, user_agent: str):
        """Register Reddit client"""
        self.clients['reddit'] = RedditClient(client_id, client_secret, user_agent)
        logger.info("Reddit client registered")
    
    def register_analysis_callback(self, callback: Callable):
        """Register callback to be called when posts are analyzed"""
        self.analysis_callbacks.append(callback)
    
    async def start_monitoring(
        self,
        platform: str,
        query: str,
        task_id: Optional[str] = None,
        poll_interval: int = 60
    ) -> str:
        """
        Start monitoring a platform for specific query.
        
        Args:
            platform: 'twitter' or 'reddit'
            query: Search query/keywords
            task_id: Optional custom task ID
            poll_interval: Seconds between polls
        
        Returns:
            task_id: ID of created monitoring task
        """
        if platform not in self.clients:
            raise ValueError(f"Platform '{platform}' not registered")
        
        if task_id is None:
            task_id = f"{platform}_{datetime.now().timestamp()}"
        
        task = MonitoringTask(
            task_id=task_id,
            platform=platform,
            query=query,
            status='running',
            started_at=datetime.now()
        )
        
        self.tasks[task_id] = task
        
        # Log task to database
        self._log_task(task)
        
        # Start monitoring in background
        asyncio.create_task(
            self._monitor_loop(task, poll_interval)
        )
        
        logger.info(f"Started monitoring: {task_id} on {platform}")
        return task_id
    
    async def _monitor_loop(self, task: MonitoringTask, poll_interval: int):
        """Continuous monitoring loop"""
        client = self.clients[task.platform]
        seen_posts = set()
        
        while task.status == 'running':
            try:
                # Fetch recent posts
                posts = await client.search_posts(task.query, limit=50)
                
                # Process new posts
                for post in posts:
                    if post.post_id not in seen_posts:
                        seen_posts.add(post.post_id)
                        
                        # Store post
                        self._store_post(post, task.query)
                        
                        # Trigger analysis callbacks
                        for callback in self.analysis_callbacks:
                            try:
                                await callback(post)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                        
                        task.posts_analyzed += 1
                
                logger.debug(
                    f"Task {task.task_id}: analyzed {task.posts_analyzed} posts"
                )
                
            except Exception as e:
                logger.error(f"Monitoring error for {task.task_id}: {e}")
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    def pause_monitoring(self, task_id: str):
        """Pause a monitoring task"""
        if task_id in self.tasks:
            self.tasks[task_id].status = 'paused'
            logger.info(f"Paused monitoring: {task_id}")
    
    def resume_monitoring(self, task_id: str):
        """Resume a monitoring task"""
        if task_id in self.tasks:
            self.tasks[task_id].status = 'running'
            logger.info(f"Resumed monitoring: {task_id}")
    
    def stop_monitoring(self, task_id: str):
        """Stop and remove a monitoring task"""
        if task_id in self.tasks:
            self.tasks[task_id].status = 'stopped'
            del self.tasks[task_id]
            logger.info(f"Stopped monitoring: {task_id}")
    
    def get_monitoring_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a monitoring task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return {
                'task_id': task.task_id,
                'platform': task.platform,
                'query': task.query,
                'status': task.status,
                'started_at': task.started_at.isoformat(),
                'posts_analyzed': task.posts_analyzed
            }
        return None
    
    def list_monitoring_tasks(self) -> List[Dict]:
        """List all active monitoring tasks"""
        return [
            {
                'task_id': task.task_id,
                'platform': task.platform,
                'query': task.query,
                'status': task.status,
                'posts_analyzed': task.posts_analyzed
            }
            for task in self.tasks.values()
        ]
    
    def _store_post(self, post: SocialPost, query: str):
        """Store monitored post in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO monitored_posts
            (id, platform, query, author, content, timestamp, url, engagement)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            post.post_id,
            post.platform,
            query,
            post.author,
            post.content,
            post.timestamp.isoformat(),
            post.url,
            str(post.engagement)
        ))
        
        conn.commit()
        conn.close()
    
    def _log_task(self, task: MonitoringTask):
        """Log monitoring task start"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO monitoring_tasks
            (task_id, platform, query, status, started_at, posts_analyzed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id,
            task.platform,
            task.query,
            task.status,
            task.started_at.isoformat(),
            task.posts_analyzed
        ))
        
        conn.commit()
        conn.close()
    
    def get_monitored_posts(
        self,
        platform: str = None,
        query: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Retrieve monitored posts from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = 'SELECT * FROM monitored_posts WHERE 1=1'
        params = []
        
        if platform:
            sql += ' AND platform = ?'
            params.append(platform)
        
        if query:
            sql += ' AND query = ?'
            params.append(query)
        
        sql += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]


# Global monitor instance
_monitor: Optional[RealtimeMonitor] = None


def get_monitor() -> RealtimeMonitor:
    """Get or create global monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = RealtimeMonitor()
    return _monitor
