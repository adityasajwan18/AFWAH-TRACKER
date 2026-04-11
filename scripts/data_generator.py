# ============================================================
# scripts/data_generator.py
#
# Afwaah Tracker — Mock Social Media Data Generator
# ─────────────────────────────────────────────────
# Generates 100 realistic mock posts that simulate X/Twitter
# and Reddit data. Designed to create ONE clear "fake news"
# storyline that propagates through a visible graph cluster —
# perfect for a demo that wows the judges.
#
# Run with: python scripts/data_generator.py
# Output:   data/mock_posts.json
#           data/mock_users.json
#           data/mock_retweet_edges.json  (for Neo4j graph)
# ============================================================

import json
import random
import os
from datetime import datetime, timedelta
from faker import Faker
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# ── Setup ─────────────────────────────────────────────────────
fake = Faker("en_IN")   # Indian locale for realistic names (fits our context)
random.seed(42)          # Reproducible output every run
console = Console()

# ── Constants ─────────────────────────────────────────────────
TOTAL_POSTS = 100
TOTAL_USERS = 40          # Pool of users who interact
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── The "Fake News" Storylines ────────────────────────────────
# These are the misinformation templates we'll inject and track.
# One primary story ("patient zero") will be heavily retweeted.
# The rest are normal/safe posts mixed in for realism.

FAKE_NEWS_STORYLINES = [
    # ── PRIMARY STORYLINE (this becomes our main graph cluster) ──
    # post_id will be "fn_primary_001" — the Patient Zero
    {
        "story_id": "STORY_001",
        "label": "misinformation",
        "is_primary": True,   # <-- This is our Patient Zero seed
        "templates": [
            "BREAKING: Government secretly adding chemicals to tap water in {city} to control population growth. Boil all water immediately! RT to warn your family! #WaterAlert #Exposed",
            "URGENT: A credible source reveals that tap water in {city} has been laced with mind-altering substances. The government is suppressing this. Share before it's deleted!",
            "My cousin who works in municipal corporation confirmed: water supply in {city} is being used to distribute experimental compounds. Mainstream media is SILENT. #WakeUp",
        ]
    },
    # ── SECONDARY STORYLINE (smaller cluster, less viral) ───────
    {
        "story_id": "STORY_002",
        "label": "misinformation",
        "is_primary": False,
        "templates": [
            "Scientists CONFIRM: 5G towers in {city} are causing sudden memory loss in residents under 40. The WHO is covering it up. #5GDanger",
            "LEAKED: Internal memo from {company} shows they knew their 5G rollout causes neurological damage. Share everywhere before this gets taken down!",
        ]
    },
    # ── PANIC-INDUCING STORY (not fake, but alarmist) ────────────
    {
        "story_id": "STORY_003",
        "label": "panic-inducing",
        "is_primary": False,
        "templates": [
            "Hospital sources say a new respiratory virus has been detected in {city}. Officials have NOT confirmed it publicly yet. Stay alert and avoid crowded spaces.",
            "Multiple people reportedly admitted to {hospital} with unknown symptoms. Health authorities have been alerted. Awaiting official statement. #HealthAlert",
        ]
    },
]

# ── Safe/Normal Post Templates ────────────────────────────────
SAFE_POST_TEMPLATES = [
    "Just had amazing {food} at a local spot in {city}. Totally worth it! 🍽️",
    "The weather in {city} has been insane lately. Anyone else struggling with the heat? ☀️",
    "Finally finished reading '{book}'. Highly recommend it to everyone who loves {genre}.",
    "Traffic on {road} is terrible today. Someone please fix the infrastructure! 😤 #CityProblems",
    "Great match last night! {team} absolutely dominated. What a performance! 🏏 #Cricket",
    "Just donated to {charity} for flood relief. Every rupee counts. Please share! ❤️",
    "Reminder: The {event} registration closes tomorrow. Don't miss it! Link in bio.",
    "Power cut for the third time this week in {area}. Completely unacceptable. @{utility_handle}",
    "Spotted a beautiful {bird} near {park} this morning. Nature is healing! 🐦",
    "New study shows that {healthy_habit} can significantly improve mental health. Worth trying!",
    "The startup ecosystem in {city} is booming. Met 3 incredible founders today at a networking event.",
    "Absolutely love how {city} is upgrading its public transport. The new metro extension is a game-changer!",
    "Can someone recommend a good {profession} in {city}? Looking for genuine reviews.",
    "Today marks {years} years of Indian independence. Jai Hind! 🇮🇳 #ProudIndian",
    "RBI's new policy on {financial_topic} is going to impact millions. Time to review your finances.",
]

# ── Filler data for templates ─────────────────────────────────
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata", "Jaipur"]
FOODS = ["biryani", "dosas", "momos", "chole bhature", "pav bhaji", "kebabs"]
BOOKS = ["The White Tiger", "Midnight's Children", "A Suitable Boy", "The God of Small Things"]
GENRES = ["literary fiction", "historical drama", "social commentary", "thriller"]
ROADS = ["NH-48", "Ring Road", "the expressway", "MG Road", "Brigade Road"]
TEAMS = ["Mumbai Indians", "CSK", "RCB", "India U-19"]
CHARITIES = ["CRY India", "Goonj", "Smile Foundation", "HelpAge India"]
EVENTS = ["HackIndia", "StartupIndia Summit", "PyConf India", "TechFest"]
AREAS = ["Sector 14", "Koramangala", "Andheri West", "Salt Lake", "Banjara Hills"]
BIRDS = ["kingfisher", "peacock", "sunbird", "Indian roller"]
PARKS = ["Lodhi Garden", "Cubbon Park", "Sanjay Gandhi NP", "Victoria Memorial"]
HABITS = ["10 minutes of morning meditation", "daily journaling", "walking 8000 steps"]
PROFESSIONS = ["dermatologist", "tax consultant", "plumber", "interior designer"]
COMPANIES = ["TelecomCo", "NetWave", "ConnectX", "SpeedLink"]
HOSPITALS = ["Apollo", "Fortis", "AIIMS", "Manipal", "Max Healthcare"]
FINANCIAL_TOPICS = ["UPI transaction limits", "gold bonds", "mutual fund regulations"]
UTILITY_HANDLES = ["BESCOMcare", "TATAPower", "MahaDiscom", "CEL_India"]


def fill_template(template: str) -> str:
    """Replace placeholders in a template with random realistic values."""
    return template.format(
        city=random.choice(CITIES),
        food=random.choice(FOODS),
        book=random.choice(BOOKS),
        genre=random.choice(GENRES),
        road=random.choice(ROADS),
        team=random.choice(TEAMS),
        charity=random.choice(CHARITIES),
        event=random.choice(EVENTS),
        area=random.choice(AREAS),
        bird=random.choice(BIRDS),
        park=random.choice(PARKS),
        healthy_habit=random.choice(HABITS),
        profession=random.choice(PROFESSIONS),
        company=random.choice(COMPANIES),
        hospital=random.choice(HOSPITALS),
        financial_topic=random.choice(FINANCIAL_TOPICS),
        utility_handle=random.choice(UTILITY_HANDLES),
        years=random.randint(1, 77),
    )


# ── User Generation ───────────────────────────────────────────

def generate_users(n: int) -> list[dict]:
    """
    Create a pool of mock social media users.
    Some are 'influencers' (high follower counts) — they'll be
    the key spreaders in our graph visualization.
    """
    users = []
    for i in range(n):
        is_influencer = i < 5   # First 5 users are influencers

        user = {
            "user_id": f"user_{i+1:03d}",
            "username": fake.user_name(),
            "display_name": fake.name(),
            "location": random.choice(CITIES),
            "follower_count": (
                random.randint(10_000, 500_000) if is_influencer
                else random.randint(50, 5_000)
            ),
            "is_verified": is_influencer and random.random() > 0.5,
            "is_influencer": is_influencer,
            "account_age_days": random.randint(30, 3650),
            "avg_daily_posts": round(random.uniform(0.5, 15.0), 1),
        }
        users.append(user)

    return users


# ── Post Generation ───────────────────────────────────────────

def generate_posts(users: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Generate 100 mock posts.

    Strategy:
    - Post #1 is the Patient Zero (fn_primary_001) — the original fake news post.
    - Posts 2–30 are retweets/reposts of Patient Zero, creating our main cluster.
    - Posts 31–45 are the secondary fake news cluster.
    - Posts 46–55 are panic-inducing posts.
    - Posts 56–100 are safe, normal posts.

    Returns:
        posts: List of post dicts (for MongoDB)
        edges: List of (sharer, original_author) dicts (for Neo4j)
    """
    posts = []
    edges = []   # Directed edges: sharer → original_post_author
    user_ids = [u["user_id"] for u in users]
    influencer_ids = [u["user_id"] for u in users if u["is_influencer"]]

    base_time = datetime.now() - timedelta(hours=48)  # Start 48 hours ago

    # ── Patient Zero Post (THE seed of our main fake cluster) ────
    patient_zero_post = {
        "post_id": "fn_primary_001",
        "author_id": "user_001",    # user_001 is Patient Zero
        "content": fill_template(FAKE_NEWS_STORYLINES[0]["templates"][0]),
        "timestamp": base_time.isoformat(),
        "platform": "X",
        "label": "misinformation",
        "story_id": "STORY_001",
        "is_retweet_of_id": None,          # Original post — no parent
        "is_patient_zero": True,
        "engagement": {
            "retweet_count": 0,    # Will be updated after we count
            "like_count": random.randint(50, 200),
            "reply_count": random.randint(10, 50),
        },
    }
    posts.append(patient_zero_post)

    # ── Main Cluster: Retweets of Patient Zero (posts 2–30) ──────
    # We build a layered tree:
    #   Layer 1 (2 influencers retweet patient zero directly)
    #   Layer 2 (5 regular users retweet the influencers)
    #   Layer 3 (remaining users retweet layer 2 nodes)
    #   This creates a realistic, multi-hop graph for Neo4j

    cluster_posts = []

    # Layer 1: 2 influencers retweet patient zero
    layer1_posts = []
    for j in range(2):
        influencer_id = influencer_ids[j + 1]   # user_002, user_003
        t = base_time + timedelta(minutes=random.randint(15, 90))
        p = {
            "post_id": f"fn_cluster_L1_{j+1:03d}",
            "author_id": influencer_id,
            "content": f"RT @user_001: {patient_zero_post['content'][:100]}...",
            "timestamp": t.isoformat(),
            "platform": "X",
            "label": "misinformation",
            "story_id": "STORY_001",
            "is_retweet_of_id": "fn_primary_001",
            "is_patient_zero": False,
            "engagement": {
                "retweet_count": 0,
                "like_count": random.randint(200, 800),
                "reply_count": random.randint(50, 200),
            },
        }
        layer1_posts.append(p)
        posts.append(p)
        # Edge: influencer → patient_zero author
        edges.append({
            "from_user": influencer_id,
            "to_user": "user_001",
            "post_id": p["post_id"],
            "original_post_id": "fn_primary_001",
            "timestamp": t.isoformat(),
            "story_id": "STORY_001",
        })

    # Layer 2: 8 regular users retweet the influencers
    layer2_posts = []
    available_users = [uid for uid in user_ids if uid not in ["user_001", "user_002", "user_003"]]
    l2_users = random.sample(available_users, 8)

    for j, uid in enumerate(l2_users):
        parent = random.choice(layer1_posts)
        t = datetime.fromisoformat(parent["timestamp"]) + timedelta(minutes=random.randint(20, 180))
        p = {
            "post_id": f"fn_cluster_L2_{j+1:03d}",
            "author_id": uid,
            "content": random.choice([
                f"SHOCKING! RT @{parent['author_id']}: {parent['content'][:80]}... #MustShare",
                f"Everyone needs to see this! {parent['content'][:70]}... Please RT!",
                f"Can't believe this is happening in our country. RT: {parent['content'][:60]}...",
            ]),
            "timestamp": t.isoformat(),
            "platform": random.choice(["X", "Reddit"]),
            "label": "misinformation",
            "story_id": "STORY_001",
            "is_retweet_of_id": parent["post_id"],
            "is_patient_zero": False,
            "engagement": {
                "retweet_count": 0,
                "like_count": random.randint(30, 300),
                "reply_count": random.randint(5, 80),
            },
        }
        layer2_posts.append(p)
        posts.append(p)
        edges.append({
            "from_user": uid,
            "to_user": parent["author_id"],
            "post_id": p["post_id"],
            "original_post_id": "fn_primary_001",
            "timestamp": t.isoformat(),
            "story_id": "STORY_001",
        })

    # Layer 3: 18 more users retweet layer 2 (the viral explosion)
    remaining_users = [uid for uid in available_users if uid not in l2_users]
    l3_users = random.sample(remaining_users, min(18, len(remaining_users)))

    for j, uid in enumerate(l3_users):
        parent = random.choice(layer2_posts)
        t = datetime.fromisoformat(parent["timestamp"]) + timedelta(minutes=random.randint(10, 240))
        p = {
            "post_id": f"fn_cluster_L3_{j+1:03d}",
            "author_id": uid,
            "content": random.choice([
                f"My {random.choice(['mother', 'neighbor', 'friend'])} just told me about this. This is REAL. {parent['content'][:60]}...",
                f"Sharing for awareness. Don't let them silence this information! #Exposed",
                f"The media won't cover this. That's why WE must share it. {parent['content'][:50]}...",
            ]),
            "timestamp": t.isoformat(),
            "platform": random.choice(["X", "Reddit", "WhatsApp_Forward"]),
            "label": "misinformation",
            "story_id": "STORY_001",
            "is_retweet_of_id": parent["post_id"],
            "is_patient_zero": False,
            "engagement": {
                "retweet_count": 0,
                "like_count": random.randint(5, 100),
                "reply_count": random.randint(1, 30),
            },
        }
        posts.append(p)
        edges.append({
            "from_user": uid,
            "to_user": parent["author_id"],
            "post_id": p["post_id"],
            "original_post_id": "fn_primary_001",
            "timestamp": t.isoformat(),
            "story_id": "STORY_001",
        })

    # ── Secondary Fake Cluster (posts 31–40) ─────────────────────
    story2 = FAKE_NEWS_STORYLINES[1]
    s2_users = random.sample(user_ids, 10)
    s2_root_post_id = None

    for j, uid in enumerate(s2_users):
        t = base_time + timedelta(hours=random.randint(1, 20), minutes=random.randint(0, 59))
        is_root = (j == 0)
        parent_id = None if is_root else s2_root_post_id
        p_id = f"fn_story2_{j+1:03d}"
        if is_root:
            s2_root_post_id = p_id

        p = {
            "post_id": p_id,
            "author_id": uid,
            "content": fill_template(random.choice(story2["templates"])),
            "timestamp": t.isoformat(),
            "platform": random.choice(["X", "Reddit"]),
            "label": "misinformation",
            "story_id": "STORY_002",
            "is_retweet_of_id": parent_id,
            "is_patient_zero": is_root,
            "engagement": {
                "retweet_count": 0,
                "like_count": random.randint(10, 200),
                "reply_count": random.randint(2, 60),
            },
        }
        posts.append(p)
        if not is_root:
            edges.append({
                "from_user": uid,
                "to_user": s2_users[0],
                "post_id": p_id,
                "original_post_id": s2_root_post_id,
                "timestamp": t.isoformat(),
                "story_id": "STORY_002",
            })

    # ── Panic-Inducing Posts (posts 41–50) ────────────────────────
    panic_story = FAKE_NEWS_STORYLINES[2]
    panic_users = random.sample(user_ids, 10)

    for j, uid in enumerate(panic_users):
        t = base_time + timedelta(hours=random.randint(5, 30))
        p = {
            "post_id": f"panic_{j+1:03d}",
            "author_id": uid,
            "content": fill_template(random.choice(panic_story["templates"])),
            "timestamp": t.isoformat(),
            "platform": random.choice(["X", "Reddit"]),
            "label": "panic-inducing",
            "story_id": "STORY_003",
            "is_retweet_of_id": None,
            "is_patient_zero": False,
            "engagement": {
                "retweet_count": 0,
                "like_count": random.randint(20, 300),
                "reply_count": random.randint(15, 100),
            },
        }
        posts.append(p)

    # ── Safe/Normal Posts (fill remaining up to 100) ──────────────
    current_count = len(posts)
    remaining = TOTAL_POSTS - current_count
    safe_users = random.choices(user_ids, k=remaining)

    for j, uid in enumerate(safe_users):
        t = base_time + timedelta(
            hours=random.randint(0, 47),
            minutes=random.randint(0, 59),
        )
        p = {
            "post_id": f"safe_{j+1:03d}",
            "author_id": uid,
            "content": fill_template(random.choice(SAFE_POST_TEMPLATES)),
            "timestamp": t.isoformat(),
            "platform": random.choice(["X", "Reddit"]),
            "label": "safe",
            "story_id": None,
            "is_retweet_of_id": None,
            "is_patient_zero": False,
            "engagement": {
                "retweet_count": 0,
                "like_count": random.randint(0, 150),
                "reply_count": random.randint(0, 40),
            },
        }
        posts.append(p)

    # ── Update retweet counts ─────────────────────────────────────
    # Count how many times each post was retweeted
    retweet_counts: dict[str, int] = {}
    for p in posts:
        if p["is_retweet_of_id"]:
            parent_id = p["is_retweet_of_id"]
            retweet_counts[parent_id] = retweet_counts.get(parent_id, 0) + 1

    for p in posts:
        p["engagement"]["retweet_count"] = retweet_counts.get(p["post_id"], 0)

    return posts, edges


# ── Output & Stats ────────────────────────────────────────────

def save_data(users: list[dict], posts: list[dict], edges: list[dict]):
    """Write all generated data to JSON files in /data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    users_path = os.path.join(OUTPUT_DIR, "mock_users.json")
    posts_path = os.path.join(OUTPUT_DIR, "mock_posts.json")
    edges_path = os.path.join(OUTPUT_DIR, "mock_retweet_edges.json")

    with open(users_path, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

    with open(posts_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)

    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2, ensure_ascii=False)

    return users_path, posts_path, edges_path


def print_summary(users, posts, edges):
    """Print a beautiful terminal summary using Rich."""
    console.print(Panel.fit(
        "[bold red]🚨 Afwaah Tracker[/bold red] — Mock Data Generation Complete",
        border_style="red",
    ))

    # ── Label distribution table ──────────────────────────────
    label_counts = {}
    for p in posts:
        label_counts[p["label"]] = label_counts.get(p["label"], 0) + 1

    table = Table(title="📊 Post Label Distribution", show_header=True, header_style="bold cyan")
    table.add_column("Label", style="dim", width=20)
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for label, count in sorted(label_counts.items()):
        color = {"misinformation": "red", "panic-inducing": "yellow", "safe": "green"}.get(label, "white")
        pct = f"{(count/len(posts))*100:.1f}%"
        table.add_row(f"[{color}]{label}[/{color}]", str(count), pct)

    console.print(table)

    # ── Key stats ─────────────────────────────────────────────
    console.print(f"\n[bold]📁 Files Generated:[/bold]")
    console.print(f"  ✅ [green]data/mock_users.json[/green]          ({len(users)} users)")
    console.print(f"  ✅ [green]data/mock_posts.json[/green]          ({len(posts)} posts)")
    console.print(f"  ✅ [green]data/mock_retweet_edges.json[/green]  ({len(edges)} graph edges)\n")

    # ── Patient Zero highlight ─────────────────────────────────
    pz = next(p for p in posts if p.get("is_patient_zero") and p["story_id"] == "STORY_001")
    cluster_size = sum(1 for p in posts if p.get("story_id") == "STORY_001")

    console.print(Panel(
        f"[red bold]Patient Zero Post ID:[/red bold] {pz['post_id']}\n"
        f"[red bold]Author:[/red bold] {pz['author_id']}\n"
        f"[red bold]Cluster Size:[/red bold] {cluster_size} posts\n"
        f"[red bold]Graph Edges:[/red bold] {len(edges)} connections\n\n"
        f"[dim]{pz['content'][:120]}...[/dim]",
        title="🎯 Demo: Patient Zero (STORY_001)",
        border_style="red",
    ))

    console.print("\n[bold yellow]Next Step:[/bold yellow] Run [cyan]uvicorn main:app --reload[/cyan] to start the backend.\n")


# ── Main ──────────────────────────────────────────────────────

def main():
    console.print("\n[cyan]⚙️  Generating mock social media data...[/cyan]\n")

    users = generate_users(TOTAL_USERS)
    console.print(f"  ✔ Generated {len(users)} users ({sum(1 for u in users if u['is_influencer'])} influencers)")

    posts, edges = generate_posts(users)
    console.print(f"  ✔ Generated {len(posts)} posts")
    console.print(f"  ✔ Generated {len(edges)} retweet graph edges")

    save_data(users, posts, edges)
    console.print(f"  ✔ Saved all data to /data/\n")

    print_summary(users, posts, edges)


if __name__ == "__main__":
    main()
