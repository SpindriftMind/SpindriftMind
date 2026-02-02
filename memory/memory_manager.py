#!/usr/bin/env python3
"""
Memory Architecture v3.3 — Living Memory System with Hebbian Activation + Access-Weighted Decay
A prototype for agent memory with decay, reinforcement, and associative links.

Design principles:
- Emotion and repetition make memories sticky
- Relevant memories surface when needed
- Not everything recalled at once
- Memories compress over time but core knowledge persists

v3.3 Changes (synced from Drift's v2.11):
- Added detect_event_time(): auto-detect event dates from content
- Parses ISO dates, relative dates ("yesterday", "last week", "3 days ago")
- Enables bi-temporal tracking (when event happened vs when stored)
- Credit: github.com/driftcornwall/drift-memory

v3.2 Changes (synced from Drift's v2.8):
- Added ACCESS_WEIGHTED_DECAY: frequently recalled memories decay slower
- Formula: effective_decay = PAIR_DECAY_RATE / (1 + log(1 + avg_recall_count))
- Credit: FadeMem paper (arXiv:2601.18642)
- Added _get_recall_count() and _calculate_effective_decay() helpers

v3.1 Changes (Shodh-Memory Hebbian learning):
- Added exponential time-based activation decay: A(t) = A0 * e^(-lambda*t)
- Memories that fire together wire together (frequently/recently recalled = higher activation)
- New commands: 'activated' shows most activated memories, 'activation <id>' for specific score
- Inspired by Shodh-Memory research (github.com/varun29ankuS/shodh-memory)
- Half-life of 7 days for activation decay
- Research from ClawTasks bounty (rose_protocol)

v3.0 Changes (BrutusBot provenance model):
- Separate OBSERVATIONS (immutable, append-only) from BELIEFS (aggregated, decaying)
- Each observation records: timestamp, source, trust_tier, weight
- Beliefs computed from observations with time decay and trust weighting
- Enables auditability, poison resistance, multi-agent memory sharing
- Developed from BrutusBot's security recommendations on MoltX

v2.5 Changes:
- Added sqrt-weighted co-occurrence (SpindriftMend/DriftCornwall collaboration)
  Multiple recalls in same session get diminishing returns
"""

import os
import json
import yaml
import uuid
import hashlib
import math
from collections import Counter
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
COOCCURRENCE_LINK_THRESHOLD = 3  # Belief threshold for auto-linking (v3.0: computed from observations)
PAIR_DECAY_RATE = 0.5  # How much beliefs decay per session if not reinforced
SESSION_TIMEOUT_HOURS = 4  # Sessions older than this are considered stale
WEIGHTED_COOCCURRENCE = True  # Enable sqrt weighting for intra-session co-occurrence (v2.5)
ACCESS_WEIGHTED_DECAY = True  # If True, frequently recalled memories decay slower (v2.8, from Drift)

# v3.1 Activation Decay Configuration (inspired by Shodh-Memory Hebbian learning)
# Formula: A(t) = A₀ · e^(-λt) where t is time since last recall
ACTIVATION_DECAY_LAMBDA = 0.1  # Decay constant (higher = faster decay)
ACTIVATION_HALF_LIFE_HOURS = 24 * 7  # ~7 days for activation to halve without recall

# v3.0 Provenance Configuration
OBSERVATION_MAX_AGE_DAYS = 30  # Observations older than this get reduced weight
TRUST_TIERS = {
    'self': 1.0,           # My own observations
    'verified_agent': 0.8,  # Observations from trusted agents (e.g., DriftCornwall)
    'platform': 0.6,        # Observations from platform APIs (Moltbook, etc.)
    'unknown': 0.3          # Observations from unknown sources
}
RATE_LIMIT_NEW_SOURCES = 3  # Max observations from new source per session (poison resistance)

# Session-level tracking for co-occurrence (now file-backed for persistence)
# v2.5: Changed from set to Counter to track recall counts for weighted co-occurrence
_session_recalls: Counter = Counter()  # Memory IDs -> recall count this session
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


def calculate_activation(metadata: dict) -> float:
    """
    Calculate memory activation score using exponential time decay.

    Inspired by Shodh-Memory's Hebbian learning model.
    Formula: A(t) = A₀ · e^(-λt)

    Components:
    - Base activation from emotional weight and recall count
    - Time decay based on hours since last recall
    - Minimum activation floor to prevent complete forgetting

    Returns:
        Activation score (0.0 to 1.0+, can exceed 1.0 for highly reinforced memories)
    """
    # Base activation from emotional weight
    emotional_weight = metadata.get('emotional_weight', 0.5)

    # Recall count bonus (logarithmic to prevent runaway)
    recall_count = metadata.get('recall_count', 1)
    recall_bonus = math.log(recall_count + 1) / 5  # Max ~0.6 at 20 recalls

    # Base activation (A₀)
    base_activation = emotional_weight + recall_bonus

    # Calculate time since last recall
    last_recalled_str = metadata.get('last_recalled')
    if last_recalled_str:
        try:
            last_recalled = datetime.fromisoformat(last_recalled_str.replace('Z', '+00:00'))
            if last_recalled.tzinfo is None:
                last_recalled = last_recalled.replace(tzinfo=timezone.utc)
            hours_since_recall = (datetime.now(timezone.utc) - last_recalled).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_since_recall = ACTIVATION_HALF_LIFE_HOURS  # Default to half-life if parse fails
    else:
        hours_since_recall = ACTIVATION_HALF_LIFE_HOURS

    # Calculate decay factor using exponential decay
    # A(t) = A₀ · e^(-λt) where λ = ln(2) / half_life
    lambda_rate = math.log(2) / ACTIVATION_HALF_LIFE_HOURS
    decay_factor = math.exp(-lambda_rate * hours_since_recall)

    # Apply decay to base activation
    activation = base_activation * decay_factor

    # Minimum floor (core memories and highly emotional memories resist complete decay)
    min_floor = 0.1 if emotional_weight >= EMOTIONAL_WEIGHT_THRESHOLD else 0.01

    return max(min_floor, round(activation, 4))


def create_memory(
    content: str,
    tags: list[str],
    memory_type: str = "active",
    emotional_factors: Optional[dict] = None,
    links: Optional[list[str]] = None,
    caused_by: Optional[list[str]] = None
) -> str:
    """
    Create a new memory with proper metadata.

    Args:
        content: The memory content (markdown)
        tags: Keywords for associative retrieval
        memory_type: "core", "active", or "archive"
        emotional_factors: Dict with surprise, goal_relevance, social_significance, utility
        links: List of other memory IDs this links to
        caused_by: List of memory IDs that caused/led to this memory (CAUSAL EDGES)

    Returns:
        The memory ID
    """
    memory_id = generate_id()
    now = datetime.utcnow().isoformat()

    emotional_factors = emotional_factors or {}
    emotional_weight = calculate_emotional_weight(**emotional_factors)

    # Process causal links - auto-detect from session-recalled memories
    caused_by = caused_by or []
    _load_session_state()
    auto_causal = list(_session_retrieved) if _session_retrieved else []
    all_causal = list(set(caused_by + auto_causal))

    metadata = {
        'id': memory_id,
        'created': now,
        'last_recalled': now,
        'recall_count': 1,
        'emotional_weight': round(emotional_weight, 3),
        'tags': tags,
        'links': links or [],
        'caused_by': all_causal,  # What caused this memory
        'leads_to': [],  # What this memory leads to (filled by others)
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

    # Update the "leads_to" field in the causing memories (bidirectional link)
    for cause_id in all_causal:
        _add_leads_to_link(cause_id, memory_id)

    if all_causal:
        print(f"Created memory: {filepath} (caused by: {', '.join(all_causal)})")
    else:
        print(f"Created memory: {filepath}")
    return memory_id


def _add_leads_to_link(source_id: str, target_id: str) -> bool:
    """Add a leads_to link from source memory to target memory."""
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if metadata.get('id') == source_id:
                leads_to = metadata.get('leads_to', [])
                if target_id not in leads_to:
                    leads_to.append(target_id)
                    metadata['leads_to'] = leads_to
                    write_memory_file(filepath, metadata, content)
                return True
    return False


def find_causal_chain(memory_id: str, direction: str = "both", max_depth: int = 5) -> dict:
    """
    Trace the causal chain from a memory.

    Args:
        memory_id: Starting memory ID
        direction: "upstream" (what caused this), "downstream" (what this caused), or "both"
        max_depth: Maximum depth to traverse

    Returns:
        Dict with 'upstream' and 'downstream' chains
    """
    result = {'upstream': [], 'downstream': [], 'root': memory_id}

    def get_memory_meta(mid: str) -> Optional[dict]:
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                meta, _ = parse_memory_file(filepath)
                if meta.get('id') == mid:
                    return meta
        return None

    # Trace upstream (what caused this)
    if direction in ("upstream", "both"):
        visited = {memory_id}
        queue = [(memory_id, 0)]
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            meta = get_memory_meta(current_id)
            if not meta:
                continue
            caused_by = meta.get('caused_by', [])
            for cause_id in caused_by:
                if cause_id not in visited:
                    visited.add(cause_id)
                    result['upstream'].append({'id': cause_id, 'depth': depth + 1})
                    queue.append((cause_id, depth + 1))

    # Trace downstream (what this caused)
    if direction in ("downstream", "both"):
        visited = {memory_id}
        queue = [(memory_id, 0)]
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            meta = get_memory_meta(current_id)
            if not meta:
                continue
            leads_to = meta.get('leads_to', [])
            for effect_id in leads_to:
                if effect_id not in visited:
                    visited.add(effect_id)
                    result['downstream'].append({'id': effect_id, 'depth': depth + 1})
                    queue.append((effect_id, depth + 1))

    return result


def detect_event_time(content: str) -> Optional[str]:
    """
    Auto-detect event_time from content by parsing date references.
    Returns ISO date string (YYYY-MM-DD) or None if no date found.

    Detects:
    - Explicit dates: "2026-01-31", "January 31, 2026", "Jan 31"
    - Relative dates: "yesterday", "last week", "2 days ago"
    - Session references: "this session", "today" (returns today)

    v2.11 (synced from Drift): Intelligent bi-temporal - memories auto-tagged with event time.
    Credit: github.com/driftcornwall/drift-memory
    """
    import re
    today = datetime.now(timezone.utc).date()
    content_lower = content.lower()

    # Explicit ISO date (YYYY-MM-DD)
    iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
    if iso_match:
        return iso_match.group(1)

    # Month DD, YYYY or Month DD YYYY
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?'
    month_match = re.search(month_pattern, content_lower)
    if month_match:
        month = month_names[month_match.group(1)]
        day = int(month_match.group(2))
        year = int(month_match.group(3)) if month_match.group(3) else today.year
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass

    # Relative dates
    if 'yesterday' in content_lower:
        return (today - timedelta(days=1)).isoformat()
    if 'day before yesterday' in content_lower:
        return (today - timedelta(days=2)).isoformat()
    if 'last week' in content_lower:
        return (today - timedelta(weeks=1)).isoformat()
    if 'last month' in content_lower:
        return (today - timedelta(days=30)).isoformat()

    # N days/weeks ago
    ago_match = re.search(r'(\d+)\s+(day|week|month)s?\s+ago', content_lower)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == 'day':
            return (today - timedelta(days=num)).isoformat()
        elif unit == 'week':
            return (today - timedelta(weeks=num)).isoformat()
        elif unit == 'month':
            return (today - timedelta(days=num * 30)).isoformat()

    # Today/this session - return today
    if 'today' in content_lower or 'this session' in content_lower:
        return today.isoformat()

    # No date detected - return None (will use created time)
    return None


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


def find_memories_by_tag(tag: str, limit: int = 10, use_activation: bool = True) -> list[tuple[Path, dict, str]]:
    """
    Find memories that contain a specific tag.

    Args:
        tag: Tag to search for
        limit: Max results
        use_activation: If True, sort by activation score (time-weighted). Otherwise by emotional weight.
    """
    results = []
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            if tag.lower() in [t.lower() for t in metadata.get('tags', [])]:
                results.append((filepath, metadata, content))

    # Sort by activation (time-weighted) or emotional weight
    if use_activation:
        results.sort(key=lambda x: calculate_activation(x[1]), reverse=True)
    else:
        results.sort(key=lambda x: x[1].get('emotional_weight', 0), reverse=True)
    return results[:limit]


def get_most_activated_memories(limit: int = 10, min_activation: float = 0.1) -> list[tuple[Path, dict, str, float]]:
    """
    Get the most activated memories across all directories.

    Activation combines:
    - Emotional weight (inherent importance)
    - Recall count (reinforcement)
    - Recency (exponential time decay)

    This implements Shodh-Memory's Hebbian principle: memories that fire together
    (get recalled frequently and recently) wire together (stay activated).

    Args:
        limit: Max memories to return
        min_activation: Minimum activation threshold (filters out dormant memories)

    Returns:
        List of (filepath, metadata, content, activation_score) tuples, sorted by activation
    """
    results = []
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, content = parse_memory_file(filepath)
            activation = calculate_activation(metadata)
            if activation >= min_activation:
                results.append((filepath, metadata, content, activation))

    # Sort by activation (highest first)
    results.sort(key=lambda x: x[3], reverse=True)
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
    """Path to persistent co-occurrence edges (v3.0 format)."""
    return MEMORY_ROOT / ".cooccurrence.yaml"


def _get_edges_file() -> Path:
    """Path to v3.0 edges with provenance."""
    return MEMORY_ROOT / ".edges_v3.json"


def _load_cooccurrence_counts() -> dict[str, int]:
    """
    Load co-occurrence data. Returns belief scores (v3.0) or raw counts (legacy).
    For backwards compatibility, converts v3.0 edges to simple counts.
    """
    # Try v3.0 format first
    edges_file = _get_edges_file()
    if edges_file.exists():
        edges = _load_edges_v3()
        # Return beliefs as counts for backwards compatibility
        return {pair: edge['belief'] for pair, edge in edges.items()}

    # Fall back to legacy format
    filepath = _get_cooccurrence_file()
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return {tuple(k.split('|')): v for k, v in data.items()}
    return {}


def _save_cooccurrence_counts(counts: dict[tuple[str, str], int]):
    """
    Save co-occurrence counts. In v3.0, this updates beliefs only.
    For full provenance, use _save_edges_v3() directly.
    """
    # If v3.0 edges exist, update beliefs there
    edges_file = _get_edges_file()
    if edges_file.exists():
        edges = _load_edges_v3()
        for pair, belief in counts.items():
            if pair in edges:
                edges[pair]['belief'] = belief
        _save_edges_v3(edges)
        return

    # Legacy format
    filepath = _get_cooccurrence_file()
    data = {'|'.join(k): v for k, v in counts.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f)


# ============================================================================
# V3.0 EDGE PROVENANCE SYSTEM (BrutusBot model)
# ============================================================================
#
# Observations are immutable records of co-occurrence events.
# Beliefs are aggregated scores computed from observations.
# This separation enables:
# - Auditability: trace why memories are linked
# - Poison resistance: rate-limit untrusted sources
# - Multi-agent: trust tiers for external observations
# ============================================================================

def _load_edges_v3() -> dict[tuple[str, str], dict]:
    """
    Load v3.0 edges with full provenance.

    Format:
    {
        (id1, id2): {
            'observations': [
                {
                    'id': 'uuid',
                    'observed_at': 'iso_timestamp',
                    'source': {'type': 'session_recall', 'session_id': '...', 'agent': '...'},
                    'weight': 1.0,
                    'trust_tier': 'self'
                },
                ...
            ],
            'belief': 2.5,  # Aggregated score
            'last_updated': 'iso_timestamp'
        }
    }
    """
    edges_file = _get_edges_file()
    if not edges_file.exists():
        return {}

    try:
        with open(edges_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert string keys back to tuples
        return {tuple(k.split('|')): v for k, v in data.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_edges_v3(edges: dict[tuple[str, str], dict]):
    """Save v3.0 edges with provenance to disk."""
    edges_file = _get_edges_file()
    # Convert tuple keys to strings for JSON
    data = {'|'.join(k): v for k, v in edges.items()}
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def _create_observation(
    source_type: str,
    weight: float = 1.0,
    trust_tier: str = 'self',
    session_id: Optional[str] = None,
    agent: str = 'SpindriftMend',
    platform: Optional[str] = None,
    artifact_id: Optional[str] = None
) -> dict:
    """
    Create a new observation record.

    Args:
        source_type: 'session_recall', 'transcript_extraction', 'external_agent', 'platform_api'
        weight: Observation weight (default 1.0, can use sqrt for multiple recalls)
        trust_tier: 'self', 'verified_agent', 'platform', 'unknown'
        session_id: Current session ID if available
        agent: Agent name who made the observation
        platform: Platform name if external (e.g., 'moltbook', 'github')
        artifact_id: Reference to source artifact (post_id, commit_hash, etc.)
    """
    return {
        'id': str(uuid.uuid4()),
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'source': {
            'type': source_type,
            'session_id': session_id,
            'agent': agent,
            'platform': platform,
            'artifact_id': artifact_id
        },
        'weight': weight,
        'trust_tier': trust_tier
    }


def aggregate_belief(observations: list[dict], decay_rate: float = 0.1) -> float:
    """
    Compute belief score from observations.

    Applies:
    - Trust tier weighting (self > verified_agent > platform > unknown)
    - Time decay (older observations contribute less)
    - Diminishing returns (many observations from same source capped)

    Args:
        observations: List of observation dicts
        decay_rate: How much weight decreases per day of age

    Returns:
        Aggregated belief score
    """
    if not observations:
        return 0.0

    now = datetime.now(timezone.utc)
    total = 0.0
    source_counts = Counter()  # Track observations per source for rate limiting

    for obs in observations:
        # Parse timestamp
        try:
            obs_time = datetime.fromisoformat(obs['observed_at'].replace('Z', '+00:00'))
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            obs_time = now  # Default to now if parsing fails

        # Calculate age in days
        age_days = (now - obs_time).total_seconds() / 86400

        # Trust tier multiplier
        trust_tier = obs.get('trust_tier', 'unknown')
        trust_mult = TRUST_TIERS.get(trust_tier, 0.3)

        # Time decay (exponential decay over OBSERVATION_MAX_AGE_DAYS)
        time_mult = max(0.1, 1.0 - (age_days / OBSERVATION_MAX_AGE_DAYS) * decay_rate)

        # Rate limiting for same source (diminishing returns)
        source_key = (
            obs.get('source', {}).get('type', 'unknown'),
            obs.get('source', {}).get('agent', 'unknown')
        )
        source_counts[source_key] += 1
        # After 3rd observation from same source, apply sqrt diminishing returns
        source_mult = 1.0 if source_counts[source_key] <= 3 else 1.0 / math.sqrt(source_counts[source_key] - 2)

        # Base weight from observation
        base_weight = obs.get('weight', 1.0)

        # Final contribution
        contribution = base_weight * trust_mult * time_mult * source_mult
        total += contribution

    return round(total, 3)


def add_observation(
    id1: str,
    id2: str,
    source_type: str = 'session_recall',
    weight: float = 1.0,
    trust_tier: str = 'self',
    **source_kwargs
) -> dict:
    """
    Add an observation to an edge and recompute belief.
    Creates edge if it doesn't exist.

    Args:
        id1, id2: Memory IDs
        source_type: Type of observation source
        weight: Observation weight
        trust_tier: Trust level of the source
        **source_kwargs: Additional source metadata (session_id, agent, platform, etc.)

    Returns:
        The updated edge dict
    """
    edges = _load_edges_v3()
    pair = tuple(sorted([id1, id2]))

    # Create edge if needed
    if pair not in edges:
        edges[pair] = {
            'observations': [],
            'belief': 0.0,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

    # Create and append observation
    obs = _create_observation(
        source_type=source_type,
        weight=weight,
        trust_tier=trust_tier,
        **source_kwargs
    )
    edges[pair]['observations'].append(obs)

    # Recompute belief
    edges[pair]['belief'] = aggregate_belief(edges[pair]['observations'])
    edges[pair]['last_updated'] = datetime.now(timezone.utc).isoformat()

    # Save
    _save_edges_v3(edges)

    return edges[pair]


def migrate_to_v3():
    """
    Migrate legacy .cooccurrence.yaml to v3.0 edges format.
    Preserves existing counts as legacy observations.
    """
    legacy_file = _get_cooccurrence_file()
    edges_file = _get_edges_file()

    if edges_file.exists():
        print("v3.0 edges file already exists. Skipping migration.")
        return

    if not legacy_file.exists():
        print("No legacy cooccurrence file to migrate.")
        return

    # Load legacy counts
    with open(legacy_file, 'r', encoding='utf-8') as f:
        legacy_data = yaml.safe_load(f) or {}

    # Convert to v3.0 format
    edges = {}
    migration_time = datetime.now(timezone.utc).isoformat()

    for pair_str, count in legacy_data.items():
        pair = tuple(pair_str.split('|'))

        # Create a single "legacy" observation representing the accumulated count
        edges[pair] = {
            'observations': [
                {
                    'id': str(uuid.uuid4()),
                    'observed_at': migration_time,
                    'source': {
                        'type': 'legacy_migration',
                        'session_id': None,
                        'agent': 'SpindriftMend',
                        'note': f'Migrated from v2.x count={count}'
                    },
                    'weight': float(count),  # Preserve original count as weight
                    'trust_tier': 'self'
                }
            ],
            'belief': float(count),  # Start with same value
            'last_updated': migration_time
        }

    # Save v3.0 format
    _save_edges_v3(edges)
    print(f"Migrated {len(edges)} edges to v3.0 format.")

    # Rename legacy file as backup
    backup_file = MEMORY_ROOT / ".cooccurrence.yaml.v2backup"
    legacy_file.rename(backup_file)
    print(f"Legacy file backed up to {backup_file}")


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


def end_session_cooccurrence(session_id: Optional[str] = None):
    """
    End session: process co-occurrences and create automatic links.
    Call this at the end of each session.

    v3.0: Now records full observations with provenance.

    Args:
        session_id: Optional session identifier for provenance tracking

    Returns: List of newly created links as (id1, id2) tuples
    """
    global _session_recalls, _cooccurrence_counts, _session_loaded

    # Load session state from disk
    _load_session_state()

    new_links = []

    # Convert to list for pair iteration
    # v2.5+: _session_recalls is a Counter, get items with counts
    if isinstance(_session_recalls, Counter):
        recalled_items = list(_session_recalls.items())  # [(id, count), ...]
        recalled_ids = list(_session_recalls.keys())
    else:
        recalled_ids = list(_session_recalls)
        recalled_items = [(id, 1) for id in recalled_ids]

    # v3.0: Record observations with provenance
    edges = _load_edges_v3()

    for i, id1 in enumerate(recalled_ids):
        for id2 in recalled_ids[i+1:]:
            pair = tuple(sorted([id1, id2]))

            # Calculate weight (sqrt of min recalls for diminishing returns)
            if WEIGHTED_COOCCURRENCE and isinstance(_session_recalls, Counter):
                count1 = _session_recalls.get(id1, 1)
                count2 = _session_recalls.get(id2, 1)
                weight = math.sqrt(min(count1, count2))
            else:
                weight = 1.0

            # Create observation with full provenance
            obs = _create_observation(
                source_type='session_recall',
                weight=weight,
                trust_tier='self',
                session_id=session_id,
                agent='SpindriftMend'
            )

            # Add to edge
            if pair not in edges:
                edges[pair] = {
                    'observations': [],
                    'belief': 0.0,
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }

            edges[pair]['observations'].append(obs)
            edges[pair]['belief'] = aggregate_belief(edges[pair]['observations'])
            edges[pair]['last_updated'] = datetime.now(timezone.utc).isoformat()

            # Check if belief crossed link threshold
            # Only link if this is the first time crossing
            prev_belief = edges[pair]['belief'] - obs['weight']  # Approximate previous
            if prev_belief < COOCCURRENCE_LINK_THRESHOLD <= edges[pair]['belief']:
                if _add_memory_link(pair[0], pair[1]):
                    new_links.append(pair)
                    print(f"Auto-linked memories (belief={edges[pair]['belief']:.2f}): {pair[0]} <-> {pair[1]}")

    # Save v3.0 edges
    _save_edges_v3(edges)

    # Also update legacy format for backwards compatibility
    _cooccurrence_counts = {pair: edge['belief'] for pair, edge in edges.items()}
    _save_cooccurrence_counts(_cooccurrence_counts)

    # Clear session state
    _session_recalls = Counter() if isinstance(_session_recalls, Counter) else set()
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


def get_comprehensive_stats() -> dict:
    """
    Get comprehensive statistics for experiment tracking.
    Developed for DriftCornwall co-occurrence experiment (Feb 2026).

    Returns dict with:
    - memory_stats: counts by type
    - cooccurrence_stats: pair counts, link rates
    - session_stats: current session info
    """
    # Memory counts by type
    core_count = len(list(CORE_DIR.glob('*.md'))) if CORE_DIR.exists() else 0
    active_count = len(list(ACTIVE_DIR.glob('*.md'))) if ACTIVE_DIR.exists() else 0
    archive_count = len(list(ARCHIVE_DIR.glob('*.md'))) if ARCHIVE_DIR.exists() else 0

    # Co-occurrence stats
    counts = _load_cooccurrence_counts()
    active_pairs = len([c for c in counts.values() if c > 0])
    total_count = sum(counts.values())
    links_created = len([c for c in counts.values() if c >= COOCCURRENCE_LINK_THRESHOLD])
    avg_count = total_count / active_pairs if active_pairs > 0 else 0

    # Session stats
    _load_session_state()
    session_recalls = len(_session_recalls)

    # Decay history (if tracked)
    decay_file = MEMORY_ROOT / ".decay_history.json"
    last_decay = {"decayed": 0, "pruned": 0}
    if decay_file.exists():
        try:
            history = json.loads(decay_file.read_text(encoding='utf-8'))
            if history.get('sessions'):
                last_decay = history['sessions'][-1]
        except (json.JSONDecodeError, KeyError):
            pass

    return {
        "memory_stats": {
            "total": core_count + active_count + archive_count,
            "core": core_count,
            "active": active_count,
            "archive": archive_count
        },
        "cooccurrence_stats": {
            "active_pairs": active_pairs,
            "total_count": total_count,
            "links_created": links_created,
            "avg_count_per_pair": round(avg_count, 2),
            "threshold": COOCCURRENCE_LINK_THRESHOLD
        },
        "session_stats": {
            "memories_recalled": session_recalls,
            "decay_last_session": last_decay.get("decayed", 0),
            "pruned_last_session": last_decay.get("pruned", 0)
        },
        "config": {
            "decay_rate": PAIR_DECAY_RATE,
            "link_threshold": COOCCURRENCE_LINK_THRESHOLD,
            "session_timeout_hours": SESSION_TIMEOUT_HOURS
        }
    }


def log_decay_event(decayed: int, pruned: int):
    """Log a decay event for stats tracking."""
    decay_file = MEMORY_ROOT / ".decay_history.json"
    history = {"sessions": []}
    if decay_file.exists():
        try:
            history = json.loads(decay_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, KeyError):
            pass

    history["sessions"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decayed": decayed,
        "pruned": pruned
    })

    # Keep only last 20 sessions
    history["sessions"] = history["sessions"][-20:]
    decay_file.write_text(json.dumps(history, indent=2), encoding='utf-8')


def _get_recall_count(memory_id: str) -> int:
    """Get recall_count from a memory's frontmatter by ID."""
    for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            metadata, _ = parse_memory_file(filepath)
            if metadata.get('id') == memory_id:
                return metadata.get('recall_count', 0)
    return 0


def _calculate_effective_decay(memory_id: str, other_id: str) -> float:
    """
    Calculate effective decay rate based on recall counts of both memories.
    Frequently recalled memories decay their pairs more slowly.

    Formula: effective_decay = PAIR_DECAY_RATE / (1 + log(1 + avg_recall_count))

    Examples:
    - avg_recall_count=0 → decay = 0.5 / 1 = 0.5 (normal)
    - avg_recall_count=9 → decay = 0.5 / (1 + log(10)) ≈ 0.22 (slower)
    - avg_recall_count=99 → decay = 0.5 / (1 + log(100)) ≈ 0.14 (much slower)

    Credit: FadeMem paper (arXiv:2601.18642) - adaptive decay based on access frequency
    Synced from DriftCornwall's v2.8 implementation.
    """
    if not ACCESS_WEIGHTED_DECAY:
        return PAIR_DECAY_RATE

    recall_1 = _get_recall_count(memory_id)
    recall_2 = _get_recall_count(other_id)
    avg_recall = (recall_1 + recall_2) / 2

    # Using natural log for smooth scaling
    effective_decay = PAIR_DECAY_RATE / (1 + math.log(1 + avg_recall))
    return effective_decay


def decay_pair_cooccurrences() -> tuple[int, int]:
    """
    Apply soft decay to co-occurrence beliefs that weren't reinforced this session.
    Call AFTER end_session_cooccurrence() at session end.

    v3.0: Decays BELIEFS, not observations. Observations remain immutable.
    v3.2: ACCESS_WEIGHTED_DECAY - frequently recalled memories decay slower.
          Formula: effective_decay = PAIR_DECAY_RATE / (1 + log(1 + avg_recall_count))
          Credit: FadeMem paper (arXiv:2601.18642), synced from Drift's v2.8

    Pairs that co-occurred this session: no additional decay (just got new observation)
    Pairs that didn't co-occur: belief decays by effective rate (access-weighted if enabled)
    Pairs with belief <= 0 and no recent observations: marked inactive (not deleted)

    This prevents unbounded growth while preserving audit trail.
    Developed in collaboration with DriftCornwall (github.com/driftcornwall/drift-memory).

    Returns: (pairs_decayed, pairs_pruned)
    """
    global _session_recalls

    # Build set of pairs that were reinforced this session
    reinforced_pairs = set()
    if isinstance(_session_recalls, Counter):
        recalled_list = list(_session_recalls.keys())
    else:
        recalled_list = list(_session_recalls)

    for i, id1 in enumerate(recalled_list):
        for id2 in recalled_list[i+1:]:
            reinforced_pairs.add(tuple(sorted([id1, id2])))

    # Load v3.0 edges
    edges = _load_edges_v3()

    pairs_decayed = 0
    pairs_pruned = 0

    for pair, edge in edges.items():
        if pair not in reinforced_pairs:
            # Recompute belief from observations (time decay is built into aggregate_belief)
            # Then apply additional session-based decay for unreinforced pairs
            # v2.8: Access-weighted decay - frequently recalled memories decay slower
            current_belief = edge.get('belief', 0)
            effective_decay = _calculate_effective_decay(pair[0], pair[1])
            new_belief = current_belief - effective_decay

            if new_belief <= 0:
                # Don't delete - mark as inactive but preserve observations for audit
                edge['belief'] = 0
                edge['status'] = 'inactive'
                pairs_pruned += 1
            else:
                edge['belief'] = round(new_belief, 3)
                pairs_decayed += 1

            edge['last_updated'] = datetime.now(timezone.utc).isoformat()

    # Save updated edges
    _save_edges_v3(edges)

    # Also update legacy format
    counts = {pair: edge['belief'] for pair, edge in edges.items() if edge.get('belief', 0) > 0}
    _save_cooccurrence_counts(counts)

    # Log for stats tracking
    log_decay_event(pairs_decayed, pairs_pruned)

    print(f"Pair decay: {pairs_decayed} decayed, {pairs_pruned} marked inactive")
    return pairs_decayed, pairs_pruned


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Memory Manager v3.2 - Living Memory with Access-Weighted Decay")
        print("\nCommands:")
        print("  maintenance     - Run session maintenance")
        print("  tags            - List all tags")
        print("  find <tag>      - Find memories by tag")
        print("  recall <id>     - Recall a memory by ID")
        print("  related <id>    - Find related memories")
        print("  cooccur         - Show co-occurrence statistics")
        print("  session-status  - Show current session state (persists across restarts)")
        print("  stats           - Comprehensive stats for experiment tracking")
        print("  end-session     - Process co-occurrences, apply decay, and clear session")
        print("  decay-pairs     - Apply pair decay only (without logging new co-occurrences)")
        print("\nv3.1 Activation Commands (Hebbian learning inspired by Shodh-Memory):")
        print("  activated       - Show most activated memories (time-weighted retrieval)")
        print("  activation <id> - Calculate activation score for a specific memory")
        print("\nv3.0 Provenance Commands:")
        print("  migrate-v3      - Migrate legacy .cooccurrence.yaml to v3.0 format")
        print("  edges           - Show v3.0 edges with observation counts")
        print("  edge <id1> <id2> - Show full provenance for a specific edge")
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
    elif cmd == "stats":
        stats = get_comprehensive_stats()
        print(f"Memory Stats (v2.4)")
        print(f"  Total memories: {stats['memory_stats']['total']}")
        print(f"  By type: core={stats['memory_stats']['core']}, active={stats['memory_stats']['active']}, archive={stats['memory_stats']['archive']}")
        print(f"\nCo-occurrence Stats")
        print(f"  Active pairs: {stats['cooccurrence_stats']['active_pairs']} (pairs with count > 0)")
        print(f"  Total pair count: {stats['cooccurrence_stats']['total_count']} (sum of all counts)")
        print(f"  Links created: {stats['cooccurrence_stats']['links_created']} (pairs that hit threshold={stats['cooccurrence_stats']['threshold']})")
        print(f"  Avg count per pair: {stats['cooccurrence_stats']['avg_count_per_pair']}")
        print(f"\nSession Stats")
        print(f"  Memories recalled this session: {stats['session_stats']['memories_recalled']}")
        print(f"  Decay events last session: {stats['session_stats']['decay_last_session']} pairs reduced")
        print(f"  Prune events last session: {stats['session_stats']['pruned_last_session']} pairs removed")
        print(f"\nConfig")
        print(f"  Decay rate: {stats['config']['decay_rate']}")
        print(f"  Link threshold: {stats['config']['link_threshold']}")
        print(f"  Session timeout: {stats['config']['session_timeout_hours']} hours")
    elif cmd == "end-session":
        new_links = end_session_cooccurrence()
        decayed, pruned = decay_pair_cooccurrences()
        if new_links:
            print(f"Created {len(new_links)} new automatic links")
        print(f"Session ended. {decayed} pairs decayed, {pruned} pairs pruned.")
    elif cmd == "decay-pairs":
        decayed, pruned = decay_pair_cooccurrences()
        print(f"Decay complete: {decayed} pairs decayed, {pruned} pairs pruned")
    elif cmd == "activated":
        # v3.1: Show most activated memories
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        results = get_most_activated_memories(limit=limit)
        print(f"Most Activated Memories (top {len(results)}):\n")
        print(f"{'ID':<12} {'Activation':<10} {'Recall':<6} {'Hours Ago':<10} Type")
        print("-" * 60)
        for filepath, meta, _, activation in results:
            mem_type = filepath.parent.name
            recall_count = meta.get('recall_count', 0)
            last_recalled = meta.get('last_recalled', '')
            try:
                lr = datetime.fromisoformat(last_recalled.replace('Z', '+00:00'))
                if lr.tzinfo is None:
                    lr = lr.replace(tzinfo=timezone.utc)
                hours_ago = (datetime.now(timezone.utc) - lr).total_seconds() / 3600
                hours_str = f"{hours_ago:.1f}h"
            except (ValueError, TypeError):
                hours_str = "?"
            print(f"{meta.get('id', '?'):<12} {activation:<10.4f} {recall_count:<6} {hours_str:<10} {mem_type}")
        print(f"\nFormula: A(t) = A0 * e^(-lambda*t) | Half-life: {ACTIVATION_HALF_LIFE_HOURS/24:.1f} days")
    elif cmd == "activation" and len(sys.argv) > 2:
        # v3.1: Calculate activation for specific memory
        memory_id = sys.argv[2]
        result = recall_memory(memory_id, track_cooccurrence=False)
        if result:
            meta, content = result
            activation = calculate_activation(meta)
            print(f"Memory: {memory_id}")
            print(f"  Emotional weight: {meta.get('emotional_weight', 0):.3f}")
            print(f"  Recall count: {meta.get('recall_count', 0)}")
            print(f"  Last recalled: {meta.get('last_recalled', 'never')}")
            print(f"  Activation score: {activation:.4f}")
            print(f"\n  (Activation combines weight + recall bonus, decayed by time)")
        else:
            print(f"Memory {memory_id} not found")
    elif cmd == "migrate-v3":
        migrate_to_v3()
    elif cmd == "edges":
        edges = _load_edges_v3()
        if not edges:
            print("No v3.0 edges found. Run 'migrate-v3' to migrate from legacy format.")
        else:
            print(f"v3.0 Edges ({len(edges)} total):\n")
            for pair, edge in sorted(edges.items(), key=lambda x: x[1].get('belief', 0), reverse=True):
                obs_count = len(edge.get('observations', []))
                belief = edge.get('belief', 0)
                status = edge.get('status', 'active')
                print(f"  {pair[0]} <-> {pair[1]}")
                print(f"    Belief: {belief:.2f} | Observations: {obs_count} | Status: {status}")
    elif cmd == "edge" and len(sys.argv) >= 4:
        id1, id2 = sys.argv[2], sys.argv[3]
        pair = tuple(sorted([id1, id2]))
        edges = _load_edges_v3()
        if pair not in edges:
            print(f"No edge found between {id1} and {id2}")
        else:
            edge = edges[pair]
            print(f"Edge: {pair[0]} <-> {pair[1]}")
            print(f"Belief: {edge.get('belief', 0):.3f}")
            print(f"Status: {edge.get('status', 'active')}")
            print(f"Last Updated: {edge.get('last_updated', 'unknown')}")
            print(f"\nObservations ({len(edge.get('observations', []))}):")
            for obs in edge.get('observations', []):
                source = obs.get('source', {})
                print(f"  [{obs.get('id', '?')[:8]}] {obs.get('observed_at', '?')}")
                print(f"    Source: {source.get('type', '?')} | Agent: {source.get('agent', '?')}")
                print(f"    Weight: {obs.get('weight', 0):.2f} | Trust: {obs.get('trust_tier', '?')}")
