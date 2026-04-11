# ============================================================
# backend/db/neo4j_client.py
#
# Neo4j Graph Database Client
# ────────────────────────────
# Handles all graph operations:
#  - Seeding mock data into Neo4j
#  - Finding all spreaders of a fake post
#  - Tracing the path back to Patient Zero
#  - Exporting graph data for D3.js visualization
#
# Gracefully degrades to in-memory fallback if Neo4j is not
# running — the demo always works.
# ============================================================

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Connection State ──────────────────────────────────────────
_driver = None
_neo4j_available = False


def get_driver():
    """Lazily connect to Neo4j. Returns None if unavailable."""
    global _driver, _neo4j_available
    if _driver is not None:
        return _driver
    try:
        from neo4j import GraphDatabase
        from backend.core.config import settings
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        _driver.verify_connectivity()
        _neo4j_available = True
        logger.info("✅ Neo4j connected.")
        return _driver
    except Exception as e:
        logger.warning(f"⚠️  Neo4j unavailable: {e}. Using in-memory fallback.")
        _neo4j_available = False
        return None


def is_available() -> bool:
    get_driver()
    return _neo4j_available


# ── Data Seeding ──────────────────────────────────────────────

def seed_graph(users: list[dict], posts: list[dict], edges: list[dict]):
    """
    Push all mock data into Neo4j.
    Creates :User and :Post nodes, and :RETWEETED relationships.
    Idempotent — safe to run multiple times (uses MERGE).
    """
    driver = get_driver()
    if not driver:
        logger.warning("Skipping Neo4j seed — not connected.")
        return False

    with driver.session() as session:
        # Clear existing data for a clean seed
        session.run("MATCH (n) DETACH DELETE n")

        # Create User nodes
        for user in users:
            session.run(
                """
                MERGE (u:User {user_id: $user_id})
                SET u.username = $username,
                    u.display_name = $display_name,
                    u.follower_count = $follower_count,
                    u.is_influencer = $is_influencer,
                    u.location = $location
                """,
                **user,
            )

        # Create Post nodes
        for post in posts:
            session.run(
                """
                MERGE (p:Post {post_id: $post_id})
                SET p.content = $content,
                    p.label = $label,
                    p.story_id = $story_id,
                    p.timestamp = $timestamp,
                    p.platform = $platform,
                    p.is_patient_zero = $is_patient_zero,
                    p.retweet_count = $retweet_count
                WITH p
                MATCH (u:User {user_id: $author_id})
                MERGE (u)-[:AUTHORED]->(p)
                """,
                post_id=post["post_id"],
                content=post["content"],
                label=post["label"],
                story_id=post.get("story_id", ""),
                timestamp=post["timestamp"],
                platform=post["platform"],
                is_patient_zero=post.get("is_patient_zero", False),
                retweet_count=post["engagement"]["retweet_count"],
                author_id=post["author_id"],
            )

        # Create RETWEETED relationships
        for edge in edges:
            session.run(
                """
                MATCH (sharer:User {user_id: $from_user})
                MATCH (original_author:User {user_id: $to_user})
                MERGE (sharer)-[:RETWEETED {
                    post_id: $post_id,
                    original_post_id: $original_post_id,
                    timestamp: $timestamp,
                    story_id: $story_id
                }]->(original_author)
                """,
                **edge,
            )

    logger.info(f"✅ Seeded {len(users)} users, {len(posts)} posts, {len(edges)} edges into Neo4j.")
    return True


# ── Cypher Queries ────────────────────────────────────────────

def find_spreaders(story_id: str = "STORY_001") -> list[dict]:
    """
    Find all users who spread a specific fake news story.
    Returns a list of user dicts sorted by follower count.
    """
    driver = get_driver()
    if not driver:
        return _fallback_spreaders(story_id)

    with driver.session() as session:
        result = session.run(
            """
            MATCH (u:User)-[r:RETWEETED]->(orig:User)
            WHERE r.story_id = $story_id
            RETURN DISTINCT
                u.user_id AS user_id,
                u.username AS username,
                u.follower_count AS follower_count,
                u.is_influencer AS is_influencer,
                count(r) AS share_count
            ORDER BY follower_count DESC
            """,
            story_id=story_id,
        )
        return [dict(record) for record in result]


def trace_patient_zero(story_id: str = "STORY_001") -> dict:
    """
    Trace the shortest path from any spreader back to the original author.
    Returns the Patient Zero user and the propagation path.
    """
    driver = get_driver()
    if not driver:
        return _fallback_patient_zero(story_id)

    with driver.session() as session:
        # Find the Patient Zero post for this story
        pz_result = session.run(
            """
            MATCH (u:User)-[:AUTHORED]->(p:Post)
            WHERE p.story_id = $story_id AND p.is_patient_zero = true
            RETURN u.user_id AS user_id, u.username AS username,
                   u.follower_count AS follower_count,
                   p.post_id AS post_id, p.content AS content,
                   p.timestamp AS timestamp
            LIMIT 1
            """,
            story_id=story_id,
        )
        pz_record = pz_result.single()
        if not pz_record:
            return {"error": "Patient zero not found for this story."}

        # Find the longest propagation chain (shows how far it spread)
        chain_result = session.run(
            """
            MATCH path = (spreader:User)-[:RETWEETED*1..5]->(origin:User)
            WHERE origin.user_id = $origin_id
            RETURN [node IN nodes(path) | node.user_id] AS chain,
                   length(path) AS hops
            ORDER BY hops DESC
            LIMIT 1
            """,
            origin_id=pz_record["user_id"],
        )
        chain_record = chain_result.single()

        return {
            "patient_zero": dict(pz_record),
            "propagation_chain": chain_record["chain"] if chain_record else [],
            "max_hops": chain_record["hops"] if chain_record else 0,
        }


def get_graph_data(story_id: Optional[str] = None) -> dict:
    """
    Export all nodes and edges in D3.js-compatible format:
    { nodes: [{id, label, ...}], links: [{source, target, ...}] }
    """
    driver = get_driver()
    if not driver:
        return _fallback_graph_data(story_id)

    with driver.session() as session:
        if story_id:
            node_query = """
                MATCH (u:User)-[r:RETWEETED]->(v:User)
                WHERE r.story_id = $story_id
                WITH collect(DISTINCT u) + collect(DISTINCT v) AS all_users
                UNWIND all_users AS u
                RETURN DISTINCT
                    u.user_id AS id, u.username AS username,
                    u.follower_count AS follower_count,
                    u.is_influencer AS is_influencer
            """
            edge_query = """
                MATCH (u:User)-[r:RETWEETED]->(v:User)
                WHERE r.story_id = $story_id
                RETURN u.user_id AS source, v.user_id AS target,
                       r.story_id AS story_id, r.timestamp AS timestamp
            """
        else:
            node_query = """
                MATCH (u:User)-[:RETWEETED]->(v:User)
                WITH collect(DISTINCT u) + collect(DISTINCT v) AS all_users
                UNWIND all_users AS u
                RETURN DISTINCT
                    u.user_id AS id, u.username AS username,
                    u.follower_count AS follower_count,
                    u.is_influencer AS is_influencer
            """
            edge_query = """
                MATCH (u:User)-[r:RETWEETED]->(v:User)
                RETURN u.user_id AS source, v.user_id AS target,
                       r.story_id AS story_id, r.timestamp AS timestamp
            """

        nodes_result = session.run(node_query, story_id=story_id or "")
        edges_result = session.run(edge_query, story_id=story_id or "")

        nodes = [dict(r) for r in nodes_result]
        links = [dict(r) for r in edges_result]

    return _enrich_graph_data(nodes, links, story_id)


# ── In-Memory Fallbacks ───────────────────────────────────────
# When Neo4j is not running, we compute the same answers directly
# from the JSON files. The demo looks identical either way.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _load(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        return json.load(f)


def _fallback_spreaders(story_id: str) -> list[dict]:
    edges = _load("mock_retweet_edges.json")
    users = {u["user_id"]: u for u in _load("mock_users.json")}
    share_counts: dict[str, int] = {}
    for e in edges:
        if e.get("story_id") == story_id:
            uid = e["from_user"]
            share_counts[uid] = share_counts.get(uid, 0) + 1
    result = []
    for uid, count in share_counts.items():
        u = users.get(uid, {})
        result.append({
            "user_id": uid,
            "username": u.get("username", uid),
            "follower_count": u.get("follower_count", 0),
            "is_influencer": u.get("is_influencer", False),
            "share_count": count,
        })
    return sorted(result, key=lambda x: x["follower_count"], reverse=True)


def _fallback_patient_zero(story_id: str) -> dict:
    posts = _load("mock_posts.json")
    users = {u["user_id"]: u for u in _load("mock_users.json")}
    edges = _load("mock_retweet_edges.json")

    pz_post = next(
        (p for p in posts if p.get("is_patient_zero") and p.get("story_id") == story_id),
        None,
    )
    if not pz_post:
        return {"error": "Patient zero not found."}

    author = users.get(pz_post["author_id"], {})
    story_edges = [e for e in edges if e.get("story_id") == story_id]

    # Build propagation chain by tracing edges
    chain = [pz_post["author_id"]]
    visited = {pz_post["author_id"]}
    current_targets = {pz_post["author_id"]}

    for _ in range(4):  # Max 4 hops
        next_level = set()
        for e in story_edges:
            if e["to_user"] in current_targets and e["from_user"] not in visited:
                next_level.add(e["from_user"])
                visited.add(e["from_user"])
        if not next_level:
            break
        chain.extend(list(next_level)[:3])  # Show up to 3 per level
        current_targets = next_level

    return {
        "patient_zero": {
            "user_id": pz_post["author_id"],
            "username": author.get("username", pz_post["author_id"]),
            "follower_count": author.get("follower_count", 0),
            "post_id": pz_post["post_id"],
            "content": pz_post["content"],
            "timestamp": pz_post["timestamp"],
        },
        "propagation_chain": chain,
        "max_hops": len(chain) - 1,
    }


def _fallback_graph_data(story_id: Optional[str]) -> dict:
    edges = _load("mock_retweet_edges.json")
    users = {u["user_id"]: u for u in _load("mock_users.json")}
    posts = _load("mock_posts.json")

    if story_id:
        edges = [e for e in edges if e.get("story_id") == story_id]

    # Collect all user IDs that appear in edges
    involved_ids = set()
    for e in edges:
        involved_ids.add(e["from_user"])
        involved_ids.add(e["to_user"])

    # Mark patient zero
    pz_author = next(
        (p["author_id"] for p in posts if p.get("is_patient_zero") and
         (story_id is None or p.get("story_id") == story_id)),
        None,
    )

    nodes = []
    for uid in involved_ids:
        u = users.get(uid, {"user_id": uid, "username": uid, "follower_count": 100})
        # Determine which stories this user spread
        user_stories = list({e["story_id"] for e in edges if e["from_user"] == uid})
        nodes.append({
            "id": uid,
            "username": u.get("username", uid),
            "follower_count": u.get("follower_count", 100),
            "is_influencer": u.get("is_influencer", False),
            "is_patient_zero": uid == pz_author,
            "stories": user_stories,
        })

    links = [
        {
            "source": e["from_user"],
            "target": e["to_user"],
            "story_id": e.get("story_id", ""),
            "timestamp": e.get("timestamp", ""),
        }
        for e in edges
    ]

    return _enrich_graph_data(nodes, links, story_id)


def _enrich_graph_data(nodes: list, links: list, story_id: Optional[str]) -> dict:
    """Add D3-friendly metadata to nodes and links."""
    # Color nodes by role
    for node in nodes:
        if node.get("is_patient_zero"):
            node["color"] = "#FF2D2D"
            node["role"] = "patient_zero"
            node["size"] = 20
        elif node.get("is_influencer"):
            node["color"] = "#FF8C00"
            node["role"] = "influencer"
            node["size"] = 14
        else:
            node["color"] = "#E84545"
            node["role"] = "spreader"
            node["size"] = 8

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "node_count": len(nodes),
            "link_count": len(links),
            "story_id": story_id,
            "source": "neo4j" if is_available() else "in_memory_fallback",
        },
    }
