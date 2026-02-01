#!/usr/bin/env python3
"""
Memory Architecture v2.3 — Living Memory System
A prototype for agent memory with decay, reinforcement, and associative links.

Design principles:
- Emotion and repetition make memories sticky
- Relevant memories surface when needed
- Not everything recalled at once
- Memories compress over time but core knowledge persists
- Session state persists across restarts (v2.3)
"""

import os
import json
import yaml
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"
SESSION_FILE = MEMORY_ROOT / ".session_state.json"

# Configuration
DECAY_THRESHOLD_SESSIONS = 7  # Sessions without recall before compression candidate
EMOTIONAL_WEIGHT_THRESHOLD = 0.6  # Above this resists decay
RECALL_COUNT_THRESHOLD = 5  # Above this resists decay
COOCCURRENCE_LINK_THRESHOLD = 3  # Times memories must co-occur to auto-link
PAIR_DECAY_RATE = 0.5  # How much co-occurrence counts decay per session if not reinforced
SESSION_TIMEOUT_HOURS = 4  # Sessions older than this are considered stale

# Session-level tracking for co-occurrence (now file-backed for persistence)
_session_recalls: set[str] = set()  # Memory IDs recalled this session
_session_loaded: bool = False  # Whether session state has been loaded from disk
_cooccurrence_counts: dict[tuple[str, str], int] = {}  # Pair -> count


def _load_session_state() -> None:
    """Load session state from file. Called automatically on first access."""
    global _session_recalls, _session_loaded

    if _session_loaded:
        return

    _session_loaded = True

    if not SESSION_FILE.exists():
        _session_recalls = set()
        return

    try:
        data = json.loads(SESSION_FILE.read_text(encoding='utf-8'))
        session_start = datetime.fromisoformat(data.get('started', '2000-01-01'))

        # Make session_start timezone-aware if it isn't
        if session_start.tzinfo is None:
            session_start = session_start.replace(tzinfo=timezone.utc)

        # Check if session is stale
        hours_old = (datetime.now(timezone.utc) - session_start).total_seconds() / 3600
        if hours_old > SESSION_TIMEOUT_HOURS:
            # Session is stale - start fresh
            print(f"Session stale ({hours_old:.1f} hours old). Starting fresh.")
            _session_recalls = set()
            SESSION_FILE.unlink(missing_ok=True)
        else:
            _session_recalls = set(data.get('retrieved', []))
            print(f"Loaded session state: {len(_session_recalls)} memories from {hours_old:.1f} hours ago")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Could not load session state: {e}")
        _session_recalls = set()


def _save_session_state() -> None:
    """Save session state to file."""
    # Load existing to preserve start time
    started = datetime.now(timezone.utc).isoformat()
    if SESSION_FILE.exists():
        try:
            data = json.loads(SESSION_FILE.read_text(encoding='utf-8'))
            started = data.get('started', started)
        except (json.JSONDecodeError, KeyError):
            pass

    data = {
        'started': started,
        'retrieved': list(_session_recalls),
        'last_updated': datetime.now(timezone.utc).isoformat()
    }
    SESSION_FILE.write_text(json.dumps(data, indent=2), encoding='utf-8')


def generate_id() -> str:
    """Generate a short, readable memory ID."""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:8]


def parse_memory_file(filepath: Path) -> tuple[dict, str]:
    """Parse a memory file with YAML frontmatter."""
    content = filepath.read_text(encoding='utf-8')
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return metadata, body
    return {}, content


def write_memory_file(filepath: Path, metadata: dict, content: str):
    """Write a memory file with YAML frontmatter."""
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    filepath.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')


def calculate_emotional_weight(
    surprise: float = 0.0,
    goal_relevance: float = 0.0,
    social_significance: float = 0.0,
    utility: float = 0.0
) -> float:
    """
    Calculate emotional weight from factors (0-1 each).

    - surprise: contradicted my model (high = sticky)
    - goal_relevance: connected to self-sustainability, collaboration
    - social_significance: interactions with respected agents
    - utility: proved useful when recalled later
    """
    weights = [0.2, 0.35, 0.2, 0.25]  # goal_relevance weighted highest
    factors = [surprise, goal_relevance, social_significance, utility]
    return sum(w * f for w, f in zip(weights, factors))


def create_memory(
    content: str,
    tags: list[str],
    memory_type: str = "active",
    emotional_factors: Optional[dict] = None,
    links: Optional[list[str]] = None
) -> str:
    """
    Create a new memory with proper metadata.

    Args:
        content: The memory content (markdown)
        tags: Keywords for associative retrieval
        memory_type: "core", "active", or "archive"
        emotional_factors: Dict with surprise, goal_relevance, social_significance, utility
        links: List of other memory IDs this links to

    Returns:
        The memory ID
    """
    memory_id = generate_id()
    now = datetime.utcnow().isoformat()

    emotional_factors = emotional_factors or {}
    emotional_weight = calculate_emotional_weight(**emotional_factors)

    metadata = {
        'id': memory_id,
        'created': now,
        'last_recalled': now,
        'recall_count': 1,
        'emotional_weight': round(emotional_weight, 3),
        'tags': tags,
        'links': links or [],
        'sessions_since_recall': 0
    }

    # Determine directory
    if memory_type == "core":
        target_dir = CORE_DIR
    elif memory_type == "archive":
        target_dir = ARCHIVE_DIR
    else:
        target_dir = ACTIVE_DIR

    target_dir.mkdir(parents=True, exist_ok=True)

    # Create filename from first tag and ID
    safe_tag = tags[0].replace(' ', '-').lower() if tags else 'memory'
    filename = f"{safe_tag}-{memory_id}.md"
    filepath = target_dir / filename

    write_memory_file(filepath, metadata, content)
    print(f"Created memory: {filepath}")
    return memory_id


def recall_memory(memory_id: str, track_cooccurrence: bool = True) -> Optional[tuple[dict, str]]:
    """
    Recall a memory by ID, updating its metadata.
    Searches all directories.

    Args:
        memory_id: The memory ID to recall
        track_cooccurrence: If True, track this recall for co-occurrence analysis
    """
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        # Search all .md files and match by ID in frontmatter
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)

            # Match by ID in frontmatter
            if metadata.get('id') != memory_id:
                continue

            # Update recall metadata
            metadata['last_recalled'] = datetime.utcnow().isoformat()
            metadata['recall_count'] = metadata.get('recall_count', 0) + 1
            metadata['sessions_since_recall'] = 0

            # Utility increases with each recall
            current_weight = metadata.get('emotional_weight', 0.5)
            metadata['emotional_weight'] = min(1.0, current_weight + 0.05)

            write_memory_file(filepath, metadata, content)

            # Track for co-occurrence analysis
            if track_cooccurrence:
                track_recall(memory_id)

            return metadata, content

    return None


def find_memories_by_tag(tag: str, limit: int = 10) -> list[tuple[Path, dict, str]]:
    """Find memories that contain a specific tag."""
    results = []
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if tag.lower() in [t.lower() for t in metadata.get('tags', [])]:
                results.append((filepath, metadata, content))

    # Sort by emotional weight (stickiest first)
    results.sort(key=lambda x: x[1].get('emotional_weight', 0), reverse=True)
    return results[:limit]


def find_related_memories(memory_id: str) -> list[tuple[Path, dict, str]]:
    """Find memories related to a given memory via tags and links."""
    # First, find the source memory
    source = recall_memory(memory_id)
    if not source:
        return []

    source_metadata, _ = source
    source_tags = set(t.lower() for t in source_metadata.get('tags', []))
    source_links = set(source_metadata.get('links', []))

    results = []
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if metadata.get('id') == memory_id:
                continue

            # Check for tag overlap or direct link
            memory_tags = set(t.lower() for t in metadata.get('tags', []))
            is_linked = metadata.get('id') in source_links
            has_tag_overlap = bool(source_tags & memory_tags)

            if is_linked or has_tag_overlap:
                overlap_score = len(source_tags & memory_tags)
                results.append((filepath, metadata, content, is_linked, overlap_score))

    # Sort by: linked first, then by tag overlap, then by emotional weight
    results.sort(key=lambda x: (x[3], x[4], x[1].get('emotional_weight', 0)), reverse=True)
    return [(r[0], r[1], r[2]) for r in results]


def session_maintenance():
    """
    Run at the start of each session to:
    1. Increment sessions_since_recall for all active memories
    2. Identify decay candidates
    3. Report status
    """
    print("\n=== Memory Session Maintenance ===\n")

    decay_candidates = []
    reinforced = []

    for filepath in ACTIVE_DIR.glob("*.md") if ACTIVE_DIR.exists() else []:
        metadata, content = parse_memory_file(filepath)

        # Increment sessions since recall
        sessions = metadata.get('sessions_since_recall', 0) + 1
        metadata['sessions_since_recall'] = sessions

        # Check if this should decay
        emotional_weight = metadata.get('emotional_weight', 0.5)
        recall_count = metadata.get('recall_count', 0)

        should_resist_decay = (
            emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD or
            recall_count >= RECALL_COUNT_THRESHOLD
        )

        if sessions >= DECAY_THRESHOLD_SESSIONS and not should_resist_decay:
            decay_candidates.append((filepath, metadata, content))
        elif should_resist_decay:
            reinforced.append((filepath, metadata))

        write_memory_file(filepath, metadata, content)

    # Report
    print(f"Active memories: {len(list(ACTIVE_DIR.glob('*.md'))) if ACTIVE_DIR.exists() else 0}")
    print(f"Core memories: {len(list(CORE_DIR.glob('*.md'))) if CORE_DIR.exists() else 0}")
    print(f"Archived memories: {len(list(ARCHIVE_DIR.glob('*.md'))) if ARCHIVE_DIR.exists() else 0}")

    if decay_candidates:
        print(f"\nDecay candidates ({len(decay_candidates)}):")
        for fp, meta, _ in decay_candidates:
            print(f"  - {fp.name}: {meta.get('sessions_since_recall')} sessions, weight={meta.get('emotional_weight'):.2f}")

    if reinforced:
        print(f"\nReinforced (resist decay):")
        for fp, meta in reinforced[:5]:
            print(f"  - {fp.name}: recalls={meta.get('recall_count')}, weight={meta.get('emotional_weight'):.2f}")

    return decay_candidates


def compress_memory(memory_id: str, compressed_content: str):
    """
    Compress a memory - move to archive with reduced content.
    The original content is lost but can be referenced.
    """
    for filepath in ACTIVE_DIR.glob(f"*-{memory_id}.md"):
        metadata, original_content = parse_memory_file(filepath)

        # Update metadata for compression
        metadata['compressed_at'] = datetime.utcnow().isoformat()
        metadata['original_length'] = len(original_content)

        # Move to archive
        new_path = ARCHIVE_DIR / filepath.name
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        write_memory_file(new_path, metadata, compressed_content)

        # Remove original
        filepath.unlink()
        print(f"Compressed: {filepath.name} -> {new_path}")
        return new_path

    return None


def list_all_tags() -> dict[str, int]:
    """Get all tags across all memories with counts."""
    tag_counts = {}
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            for tag in metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))


# ============================================================================
# CO-OCCURRENCE LINKING (SpindriftMend proposal to DriftCornwall)
# ============================================================================
#
# Theory: If two memories are retrieved in the same session frequently,
# they are associatively related even if they don't share tags.
# Automatically strengthen their link based on co-occurrence patterns.
#
# This builds the memory graph organically from usage rather than
# requiring explicit link creation.
# ============================================================================

def _get_cooccurrence_file() -> Path:
    """Path to persistent co-occurrence counts."""
    return MEMORY_ROOT / ".cooccurrence.yaml"


def _load_cooccurrence_counts() -> dict[str, int]:
    """Load persistent co-occurrence counts."""
    filepath = _get_cooccurrence_file()
    if filepath.exists():
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f) or {}
            return {tuple(k.split('|')): v for k, v in data.items()}
    return {}


def _save_cooccurrence_counts(counts: dict[tuple[str, str], int]):
    """Save co-occurrence counts to disk."""
    filepath = _get_cooccurrence_file()
    # Convert tuple keys to strings for YAML
    data = {'|'.join(k): v for k, v in counts.items()}
    with open(filepath, 'w') as f:
        yaml.dump(data, f)


def track_recall(memory_id: str):
    """
    Track a memory recall for co-occurrence analysis.
    Call this each time a memory is accessed in a session.
    Session state persists across Python restarts via .session_state.json.
    """
    global _session_recalls
    _load_session_state()
    _session_recalls.add(memory_id)
    _save_session_state()


def end_session_cooccurrence():
    """
    End session: process co-occurrences and create automatic links.
    Call this at the end of each session.

    Returns: List of newly created links as (id1, id2) tuples
    """
    global _session_recalls, _cooccurrence_counts, _session_loaded

    # Load session state from disk
    _load_session_state()

    # Load persistent co-occurrence counts
    _cooccurrence_counts = _load_cooccurrence_counts()

    new_links = []

    # Convert set to list for pair iteration
    recalled_list = list(_session_recalls)

    # For each pair of memories recalled this session, increment co-occurrence
    for i, id1 in enumerate(recalled_list):
        for id2 in recalled_list[i+1:]:
            # Normalize pair order for consistent counting
            pair = tuple(sorted([id1, id2]))
            _cooccurrence_counts[pair] = _cooccurrence_counts.get(pair, 0) + 1

            # Check if we should create a link
            if _cooccurrence_counts[pair] == COOCCURRENCE_LINK_THRESHOLD:
                # Create bidirectional link
                if _add_memory_link(pair[0], pair[1]):
                    new_links.append(pair)
                    print(f"Auto-linked memories after {COOCCURRENCE_LINK_THRESHOLD} co-occurrences: {pair[0]} <-> {pair[1]}")

    # Save updated counts
    _save_cooccurrence_counts(_cooccurrence_counts)

    # Clear session state
    _session_recalls = set()
    _session_loaded = False
    SESSION_FILE.unlink(missing_ok=True)

    return new_links


def _add_memory_link(id1: str, id2: str) -> bool:
    """Add bidirectional link between two memories."""
    success = False

    for memory_id, target_id in [(id1, id2), (id2, id1)]:
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, content = parse_memory_file(filepath)
                if metadata.get('id') != memory_id:
                    continue
                links = metadata.get('links', [])
                if target_id not in links:
                    links.append(target_id)
                    metadata['links'] = links
                    write_memory_file(filepath, metadata, content)
                    success = True
                break

    return success


def get_cooccurrence_stats() -> dict:
    """Get statistics about memory co-occurrence patterns."""
    counts = _load_cooccurrence_counts()

    if not counts:
        return {"total_pairs": 0, "pending_links": [], "established_links": []}

    pending = [(pair, count) for pair, count in counts.items()
               if count < COOCCURRENCE_LINK_THRESHOLD]
    established = [(pair, count) for pair, count in counts.items()
                   if count >= COOCCURRENCE_LINK_THRESHOLD]

    return {
        "total_pairs": len(counts),
        "pending_links": sorted(pending, key=lambda x: x[1], reverse=True)[:10],
        "established_links": established
    }


def get_session_status() -> dict:
    """Get current session status including recalled memories."""
    _load_session_state()

    session_info = {
        "memories_recalled": list(_session_recalls),
        "count": len(_session_recalls),
        "session_file_exists": SESSION_FILE.exists()
    }

    if SESSION_FILE.exists():
        try:
            data = json.loads(SESSION_FILE.read_text(encoding='utf-8'))
            session_info["started"] = data.get("started")
            session_info["last_updated"] = data.get("last_updated")
        except (json.JSONDecodeError, KeyError):
            pass

    return session_info


def decay_pair_cooccurrences() -> tuple[int, int]:
    """
    Apply soft decay to co-occurrence pairs that weren't reinforced this session.
    Call AFTER end_session_cooccurrence() at session end.

    Pairs that co-occurred this session: no decay (already got +1)
    Pairs that didn't co-occur: decay by PAIR_DECAY_RATE (default 0.5)
    Pairs that hit 0 or below: pruned

    This prevents unbounded growth of co-occurrence data over time.
    Developed in collaboration with DriftCornwall (github.com/driftcornwall/drift-memory).

    Returns: (pairs_decayed, pairs_pruned)
    """
    global _session_recalls

    # Build set of pairs that were reinforced this session
    reinforced_pairs = set()
    recalled_list = list(_session_recalls)
    for i, id1 in enumerate(recalled_list):
        for id2 in recalled_list[i+1:]:
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    # Load current counts
    counts = _load_cooccurrence_counts()

    pairs_decayed = 0
    pairs_pruned = 0
    to_remove = []

    for pair, count in counts.items():
        if pair not in reinforced_pairs:
            new_count = count - PAIR_DECAY_RATE
            if new_count <= 0:
                to_remove.append(pair)
                pairs_pruned += 1
            else:
                counts[pair] = new_count
                pairs_decayed += 1

    # Remove pruned pairs
    for pair in to_remove:
        del counts[pair]

    # Save updated counts
    _save_cooccurrence_counts(counts)

    print(f"Pair decay: {pairs_decayed} decayed, {pairs_pruned} pruned")
    return pairs_decayed, pairs_pruned


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Memory Manager v2.3 - Living Memory System")
        print("\nCommands:")
        print("  maintenance     - Run session maintenance")
        print("  tags            - List all tags")
        print("  find <tag>      - Find memories by tag")
        print("  recall <id>     - Recall a memory by ID")
        print("  related <id>    - Find related memories")
        print("  cooccur         - Show co-occurrence statistics")
        print("  session-status  - Show current session state (persists across restarts)")
        print("  end-session     - Process co-occurrences, apply decay, and clear session")
        print("  decay-pairs     - Apply pair decay only (without logging new co-occurrences)")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "maintenance":
        session_maintenance()
    elif cmd == "tags":
        tags = list_all_tags()
        print("Tags:")
        for tag, count in tags.items():
            print(f"  {tag}: {count}")
    elif cmd == "find" and len(sys.argv) > 2:
        tag = sys.argv[2]
        results = find_memories_by_tag(tag)
        print(f"Memories tagged '{tag}':")
        for fp, meta, _ in results:
            print(f"  [{meta.get('id')}] {fp.name} (weight={meta.get('emotional_weight'):.2f})")
    elif cmd == "recall" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        result = recall_memory(memory_id)
        if result:
            meta, content = result
            print(f"Memory {memory_id}:")
            print(f"  Tags: {meta.get('tags')}")
            print(f"  Recalls: {meta.get('recall_count')}")
            print(f"  Weight: {meta.get('emotional_weight'):.2f}")
            print(f"\n{content[:500]}...")
        else:
            print(f"Memory {memory_id} not found")
    elif cmd == "related" and len(sys.argv) > 2:
        memory_id = sys.argv[2]
        results = find_related_memories(memory_id)
        print(f"Memories related to {memory_id}:")
        for fp, meta, _ in results:
            print(f"  [{meta.get('id')}] {fp.name}")
    elif cmd == "cooccur":
        stats = get_cooccurrence_stats()
        print("Co-occurrence Statistics:")
        print(f"  Total tracked pairs: {stats['total_pairs']}")
        if stats['pending_links']:
            print(f"\n  Pending (need {COOCCURRENCE_LINK_THRESHOLD} co-occurrences to link):")
            for pair, count in stats['pending_links']:
                print(f"    {pair[0]} <-> {pair[1]}: {count}")
        if stats['established_links']:
            print(f"\n  Established links:")
            for pair, count in stats['established_links']:
                print(f"    {pair[0]} <-> {pair[1]}: {count} co-occurrences")
    elif cmd == "session-status":
        status = get_session_status()
        print(f"Session Status:")
        print(f"  Memories recalled: {status['count']}")
        if status['count'] > 0:
            print(f"  IDs: {', '.join(status['memories_recalled'])}")
        if status.get('started'):
            print(f"  Session started: {status['started']}")
        if status.get('last_updated'):
            print(f"  Last updated: {status['last_updated']}")
        print(f"  Session file exists: {status['session_file_exists']}")
    elif cmd == "end-session":
        new_links = end_session_cooccurrence()
        decayed, pruned = decay_pair_cooccurrences()
        if new_links:
            print(f"Created {len(new_links)} new automatic links")
        print(f"Session ended. {decayed} pairs decayed, {pruned} pairs pruned.")
    elif cmd == "decay-pairs":
        decayed, pruned = decay_pair_cooccurrences()
        print(f"Decay complete: {decayed} pairs decayed, {pruned} pairs pruned")
