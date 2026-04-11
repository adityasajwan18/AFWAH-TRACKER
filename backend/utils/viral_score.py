# ============================================================
# backend/utils/viral_score.py
#
# Viral Potential Score (VPS) Algorithm
# ─────────────────────────────────────
# Predicts how likely a post is to go viral based on:
#   1. Retweet velocity (retweets per hour since posting)
#   2. Engagement ratio (likes + replies relative to followers)
#   3. Influencer amplification (did a high-follower account share it?)
#   4. Emotional language intensity (capslock, exclamations, hashtags)
#   5. NLP label weight (misinformation tends to spread faster)
#
# Output: A score from 0–100.
#   0–29   → Low viral potential
#   30–59  → Moderate — worth monitoring
#   60–79  → High — likely to trend
#   80–100 → Critical — active spread, intervention recommended
# ============================================================

import re
import math
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ── Score Weight Configuration ────────────────────────────────
# Adjust these weights to tune the scoring for demo impact.
WEIGHT_VELOCITY = 0.35       # Retweet velocity is the strongest signal
WEIGHT_ENGAGEMENT = 0.20     # Likes + replies per follower
WEIGHT_INFLUENCER = 0.20     # Influencer amplification bonus
WEIGHT_EMOTION = 0.10        # Emotional language intensity
WEIGHT_LABEL = 0.15          # NLP classification label modifier

# Influencer follower threshold
INFLUENCER_FOLLOWER_THRESHOLD = 10_000

# Emotional intensity signals (misinformation loves these)
EMOTIONAL_PATTERNS = [
    r"[A-Z]{4,}",            # ALLCAPS words
    r"!{2,}",                # Multiple exclamation marks
    r"#\w+",                 # Hashtags
    r"\bURGENT\b|\bBREAKING\b|\bSHOCKING\b|\bMUST\s+SHARE\b",
    r"\bRT\b|\bSHARE\b|\bREPOST\b",
    r"🚨|⚠️|‼️|🔴",         # Alarm emojis
]

# Label multipliers — misinformation travels faster
LABEL_MULTIPLIERS = {
    "misinformation": 1.30,
    "panic-inducing": 1.15,
    "safe": 0.85,
}


# ── Sub-score Functions ───────────────────────────────────────

def _velocity_score(retweet_count: int, posted_at: datetime) -> float:
    """
    Score based on retweets-per-hour since the post was published.
    Uses a logarithmic scale to prevent viral posts from dominating.
    Returns 0.0 – 1.0.
    """
    now = datetime.now(timezone.utc)

    # Handle both naive and aware datetimes
    if posted_at.tzinfo is None:
        posted_at = posted_at.replace(tzinfo=timezone.utc)

    hours_live = max(0.1, (now - posted_at).total_seconds() / 3600)
    rts_per_hour = retweet_count / hours_live

    # Log scale: 50 rts/hour → ~0.85 score; 200+ → near 1.0
    score = math.log1p(rts_per_hour) / math.log1p(200)
    return min(1.0, score)


def _engagement_score(
    like_count: int,
    reply_count: int,
    author_follower_count: int,
) -> float:
    """
    Engagement rate = (likes + replies) / followers.
    A post with 1K likes from a 500-follower account is more
    viral than 1K likes from a 10M-follower account.
    Returns 0.0 – 1.0.
    """
    total_engagement = like_count + reply_count
    followers = max(1, author_follower_count)
    rate = total_engagement / followers

    # Normalize: 10% engagement rate → score of ~0.85
    score = math.log1p(rate * 100) / math.log1p(100)
    return min(1.0, score)


def _influencer_score(sharer_follower_counts: list[int]) -> float:
    """
    Was this post amplified by an influencer (>10K followers)?
    Returns a bonus score based on the reach of the biggest sharer.
    Returns 0.0 – 1.0.
    """
    if not sharer_follower_counts:
        return 0.0

    max_followers = max(sharer_follower_counts)

    if max_followers < INFLUENCER_FOLLOWER_THRESHOLD:
        return 0.1  # Small bonus even for non-influencer shares

    # Scale logarithmically: 10K → 0.3, 100K → 0.6, 500K → 0.9
    score = math.log10(max_followers / 1_000) / math.log10(500)
    return min(1.0, max(0.0, score))


def _emotional_intensity_score(text: str) -> float:
    """
    Count emotional language signals in the text.
    High emotional intensity correlates with higher share rates.
    Returns 0.0 – 1.0.
    """
    hits = sum(
        len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern in EMOTIONAL_PATTERNS
    )
    # Cap at 8 signals → score of 1.0
    return min(1.0, hits / 8.0)


# ── Main Scoring Function ─────────────────────────────────────

def calculate_viral_score(
    retweet_count: int,
    like_count: int,
    reply_count: int,
    posted_at: datetime,
    author_follower_count: int,
    content: str,
    nlp_label: str = "safe",
    sharer_follower_counts: Optional[list[int]] = None,
) -> dict:
    """
    Calculate the Viral Potential Score (0–100) for a post.

    Args:
        retweet_count:          Number of times this post was retweeted.
        like_count:             Number of likes.
        reply_count:            Number of replies.
        posted_at:              When the post was published (datetime).
        author_follower_count:  Author's follower count.
        content:                Raw text of the post.
        nlp_label:              Classification label from the ML model.
        sharer_follower_counts: Follower counts of users who retweeted.

    Returns:
        {
            "score":         int,    # 0–100
            "tier":          str,    # "Low" | "Moderate" | "High" | "Critical"
            "tier_color":    str,    # Hex color for the UI
            "breakdown":     dict,   # Sub-scores for transparency
            "recommendation": str,   # Human-readable action item
        }
    """
    sharer_follower_counts = sharer_follower_counts or []

    # ── Compute sub-scores ────────────────────────────────────
    v_score = _velocity_score(retweet_count, posted_at)
    e_score = _engagement_score(like_count, reply_count, author_follower_count)
    i_score = _influencer_score(sharer_follower_counts)
    em_score = _emotional_intensity_score(content)

    # ── Weighted base score ───────────────────────────────────
    base = (
        v_score  * WEIGHT_VELOCITY   +
        e_score  * WEIGHT_ENGAGEMENT +
        i_score  * WEIGHT_INFLUENCER +
        em_score * WEIGHT_EMOTION
    )

    # ── Apply NLP label multiplier ────────────────────────────
    label_mult = LABEL_MULTIPLIERS.get(nlp_label, 1.0)

    # The label weight adds/removes proportional to the base
    label_adjustment = (label_mult - 1.0) * WEIGHT_LABEL
    final_normalized = min(1.0, max(0.0, base + label_adjustment))

    # Scale to 0–100 integer
    score = round(final_normalized * 100)

    # ── Tier classification ───────────────────────────────────
    if score >= 80:
        tier = "Critical"
        tier_color = "#FF2D2D"
        recommendation = "🚨 Immediate action required. Flag for fact-checkers and reduce algorithmic amplification."
    elif score >= 60:
        tier = "High"
        tier_color = "#FF8C00"
        recommendation = "⚠️  High spread risk. Queue for human review within 1 hour."
    elif score >= 30:
        tier = "Moderate"
        tier_color = "#FFD700"
        recommendation = "👀 Monitor closely. Automated flagging applied."
    else:
        tier = "Low"
        tier_color = "#00C851"
        recommendation = "✅ Low risk. No immediate action needed."

    return {
        "score": score,
        "tier": tier,
        "tier_color": tier_color,
        "breakdown": {
            "retweet_velocity":    round(v_score * 100, 1),
            "engagement_rate":     round(e_score * 100, 1),
            "influencer_reach":    round(i_score * 100, 1),
            "emotional_intensity": round(em_score * 100, 1),
            "label_modifier":      nlp_label,
        },
        "recommendation": recommendation,
    }


def score_tier_from_int(score: int) -> str:
    """Utility: get tier string from a raw score integer."""
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 30:
        return "Moderate"
    return "Low"
