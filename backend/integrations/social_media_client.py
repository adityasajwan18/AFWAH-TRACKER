# ============================================================
# backend/integrations/social_media_client.py
#
# Social Media API Integration (Twitter/Reddit)
# Real-time monitoring of posts for misinformation detection
# ============================================================

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SocialPost(BaseModel):
    """Standardized social media post format"""
    platform: str  # 'twitter', 'reddit'
    post_id: str
    author: str
    content: str
    timestamp: datetime
    url: str
    engagement: Dict[str, int]  # likes, retweets, comments, etc.
    media_urls: List[str] = []


class SocialMediaClient(ABC):
    """Abstract base for social media clients"""
    
    @abstractmethod
    async def search_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        pass
    
    @abstractmethod
    async def get_trending(self, limit: int = 50) -> List[SocialPost]:
        pass
    
    @abstractmethod
    async def monitor_stream(self, query: str) -> None:
        pass


class TwitterClient(SocialMediaClient):
    """Twitter API v2 Client"""
    
    def __init__(self, bearer_token: str):
        """
        Initialize Twitter client with Bearer token.
        Get token from: https://developer.twitter.com/en/portal/dashboard
        """
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.headers = self._get_headers()
        self.monitoring = False
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "AfwaahTracker/1.0"
        }
    
    async def search_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        """
        Search Twitter for posts matching query.
        Requires academic or enterprise API access.
        """
        posts = []
        
        params = {
            'query': query,
            'max_results': min(limit, 100),
            'tweet.fields': 'created_at,author_id,public_metrics',
            'expansions': 'author_id,attachments.media_keys',
            'user.fields': 'username,verified',
            'media.fields': 'url,type'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/tweets/search/recent",
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = self._parse_tweets(data)
                    else:
                        logger.error(f"Twitter API error: {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to search Twitter: {e}")
        
        return posts
    
    async def get_trending(self, limit: int = 50) -> List[SocialPost]:
        """Get trending topics and posts"""
        params = {
            'max_results': min(limit, 100),
            'tweet.fields': 'created_at,public_metrics',
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get trending topics (requires special endpoint)
                async with session.get(
                    f"{self.base_url}/tweets/search/recent?query=lang:en",
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_tweets(data)
        
        except Exception as e:
            logger.error(f"Failed to get trending: {e}")
        
        return []
    
    async def monitor_stream(self, query: str) -> None:
        """
        Monitor live Twitter stream (requires connection upgrade).
        Streams are real-time for enterprise users.
        """
        self.monitoring = True
        params = {
            'tweet.fields': 'created_at,author_id,public_metrics',
            'expansions': 'author_id',
            'user.fields': 'username'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/tweets/search/stream",
                    headers=self.headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=None)
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if not self.monitoring:
                                break
                            if line:
                                logger.info(f"Stream data: {line}")
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
        
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """Stop stream monitoring"""
        self.monitoring = False
    
    def _parse_tweets(self, data: Dict[str, Any]) -> List[SocialPost]:
        """Parse Twitter API response into SocialPost objects"""
        posts = []
        
        if 'data' not in data:
            return posts
        
        users_map = {}
        if 'includes' in data and 'users' in data['includes']:
            users_map = {u['id']: u['username'] for u in data['includes']['users']}
        
        for tweet in data['data']:
            try:
                post = SocialPost(
                    platform='twitter',
                    post_id=tweet['id'],
                    author=users_map.get(tweet['author_id'], 'Unknown'),
                    content=tweet['text'],
                    timestamp=datetime.fromisoformat(
                        tweet['created_at'].replace('Z', '+00:00')
                    ),
                    url=f"https://twitter.com/i/web/status/{tweet['id']}",
                    engagement={
                        'likes': tweet.get('public_metrics', {}).get('like_count', 0),
                        'retweets': tweet.get('public_metrics', {}).get('retweet_count', 0),
                        'replies': tweet.get('public_metrics', {}).get('reply_count', 0),
                    }
                )
                posts.append(post)
            except Exception as e:
                logger.error(f"Failed to parse tweet: {e}")
        
        return posts


class RedditClient(SocialMediaClient):
    """Reddit API Client"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit client with credentials.
        Get from: https://www.reddit.com/prefs/apps
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.base_url = "https://oauth.reddit.com"
        self.access_token = None
        self.token_expiry = None
    
    async def authenticate(self) -> bool:
        """Get OAuth2 access token"""
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        data = {'grant_type': 'client_credentials'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://www.reddit.com/api/v1/access_token',
                    auth=auth,
                    data=data,
                    headers={'User-Agent': self.user_agent},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data['access_token']
                        return True
        
        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")
        
        return False
    
    async def search_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Search Reddit posts"""
        if not self.access_token:
            await self.authenticate()
        
        posts = []
        params = {
            'q': query,
            'limit': min(limit, 100),
            'sort': 'relevance',
            'type': 'link,self'
        }
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'User-Agent': self.user_agent
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'{self.base_url}/search',
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = self._parse_posts(data)
        
        except Exception as e:
            logger.error(f"Failed to search Reddit: {e}")
        
        return posts
    
    async def get_trending(self, limit: int = 50) -> List[SocialPost]:
        """Get trending Reddit posts"""
        if not self.access_token:
            await self.authenticate()
        
        posts = []
        params = {'limit': min(limit, 100)}
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'User-Agent': self.user_agent
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'{self.base_url}/r/all/hot',
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = self._parse_posts(data)
        
        except Exception as e:
            logger.error(f"Failed to get Reddit trending: {e}")
        
        return posts
    
    async def monitor_stream(self, query: str) -> None:
        """Monitor subreddit stream (polling-based)"""
        # Reddit doesn't have true streaming; use polling
        while True:
            posts = await self.search_posts(query, limit=50)
            for post in posts:
                logger.info(f"New Reddit post: {post.content[:100]}")
            
            await asyncio.sleep(60)  # Poll every minute
    
    def _parse_posts(self, data: Dict[str, Any]) -> List[SocialPost]:
        """Parse Reddit API response"""
        posts = []
        
        if 'data' not in data or 'children' not in data['data']:
            return posts
        
        for item in data['data']['children']:
            try:
                post_data = item['data']
                post = SocialPost(
                    platform='reddit',
                    post_id=post_data['id'],
                    author=post_data['author'],
                    content=post_data.get('title', '') + ' ' + post_data.get('selftext', ''),
                    timestamp=datetime.fromtimestamp(post_data['created_utc']),
                    url=f"https://reddit.com{post_data['permalink']}",
                    engagement={
                        'upvotes': post_data.get('ups', 0),
                        'downvotes': post_data.get('downs', 0),
                        'comments': post_data.get('num_comments', 0),
                    }
                )
                posts.append(post)
            except Exception as e:
                logger.error(f"Failed to parse Reddit post: {e}")
        
        return posts
