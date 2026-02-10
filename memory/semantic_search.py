#!/usr/bin/env python3
"""
Semantic Search for SpindriftMend's Memory System

Enables natural language queries like "what do I know about bounties?"
instead of requiring exact memory IDs.

Supports:
- OpenAI embeddings (requires OPENAI_API_KEY)
- Local models via HTTP endpoint (for Docker-based free option)

Usage:
    python semantic_search.py index          # Build/rebuild index
    python semantic_search.py search "query" # Search memories
    python semantic_search.py status         # Check index status

Credit: Adapted from DriftCornwall's drift-memory (February 2026)
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
import math

# Load environment variables from ~/.claude/.env
try:
    from dotenv import load_dotenv
    env_file = Path.home() / ".claude" / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # dotenv is optional

MEMORY_DIR = Path(__file__).parent
ACTIVE_DIR = MEMORY_DIR / "active"
CORE_DIR = MEMORY_DIR / "core"
EMBEDDINGS_FILE = MEMORY_DIR / "embeddings.json"

# Embedding dimensions vary by model:
# - OpenAI text-embedding-3-small: 1536
# - Qwen3-Embedding-8B: 4096
# We don't enforce dimension - just compare what we have

# === RESOLUTION BOOSTING (v2.19 from DriftCornwall) ===
# Memories tagged as resolution/fix/solution get score boost so solutions
# surface before problem descriptions when searching.
RESOLUTION_TAGS = {'resolution', 'procedural', 'fix', 'solution', 'howto', 'api', 'endpoint'}
RESOLUTION_BOOST = 1.25  # 25% score boost for resolution memories


def get_embedding_openai(text: str, model: str = "text-embedding-3-small") -> Optional[list[float]]:
    """Get embedding from OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import urllib.request
        import urllib.error

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = json.dumps({
            "input": text[:8000],  # Truncate to avoid token limits
            "model": model
        }).encode('utf-8')

        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['data'][0]['embedding']
    except Exception as e:
        print(f"OpenAI embedding error: {e}", file=sys.stderr)
        return None


def get_embedding_local(text: str, endpoint: str = "http://localhost:8080/embed") -> Optional[list[float]]:
    """
    Get embedding from local model endpoint.
    Supports both TEI format ({"inputs": "..."}) and generic format ({"text": "..."}).
    """
    try:
        import urllib.request

        # Try TEI format first (Hugging Face text-embeddings-inference)
        data = json.dumps({"inputs": text[:4000]}).encode('utf-8')
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            # TEI returns [[0.1, 0.2, ...]] for single input
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0]
                return result
            # Generic format
            return result.get('embedding') or result.get('embeddings', [[]])[0]
    except Exception as e:
        return None


def get_embedding(text: str) -> Optional[list[float]]:
    """
    Get embedding using best available method.
    Priority: Local (free) > OpenAI (paid)
    """
    # Check for local endpoint first (free, private)
    local_endpoint = os.getenv("LOCAL_EMBEDDING_ENDPOINT", "").strip()
    if not local_endpoint:
        # Default to localhost if docker service might be running
        local_endpoint = "http://localhost:8080/embed"

    # Try local first
    emb = get_embedding_local(text, local_endpoint)
    if emb:
        return emb

    # Fall back to OpenAI if local unavailable
    return get_embedding_openai(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def load_embeddings() -> dict:
    """Load embeddings index from disk."""
    if EMBEDDINGS_FILE.exists():
        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"memories": {}, "model": "text-embedding-3-small"}


def save_embeddings(data: dict):
    """Save embeddings index to disk."""
    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def parse_memory_file(path: Path) -> tuple[Optional[str], Optional[str]]:
    """Parse a memory file and return (id, content)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                import re
                # Extract ID from frontmatter
                id_match = re.search(r'^id:\s*(.+)$', parts[1], re.MULTILINE)
                memory_id = id_match.group(1).strip() if id_match else path.stem
                # Content is everything after frontmatter
                body = parts[2].strip()
                return memory_id, body

        return path.stem, content
    except Exception as e:
        return None, None


def load_memory_tags(memory_id: str) -> Optional[list[str]]:
    """Load tags from a memory file's YAML frontmatter."""
    import re

    # Search in active and core directories
    for directory in [ACTIVE_DIR, CORE_DIR]:
        if not directory.exists():
            continue
        for path in directory.glob("*.md"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.startswith('---'):
                    continue

                parts = content.split('---', 2)
                if len(parts) < 3:
                    continue

                frontmatter = parts[1]

                # Check if this is the right memory
                id_match = re.search(r'^id:\s*[\'"]?([^\'"]+)[\'"]?\s*$', frontmatter, re.MULTILINE)
                if not id_match or id_match.group(1).strip() != memory_id:
                    continue

                # Extract tags - handle both list and inline formats
                # List format:
                # tags:
                # - tag1
                # - tag2
                tags_section = re.search(r'^tags:\s*\n((?:\s*-\s*.+\n?)+)', frontmatter, re.MULTILINE)
                if tags_section:
                    tags = re.findall(r'-\s*(.+)', tags_section.group(1))
                    return [t.strip().strip('"\'') for t in tags]

                # Inline format: tags: [tag1, tag2] or tags: tag1, tag2
                inline_match = re.search(r'^tags:\s*\[?([^\]\n]+)\]?\s*$', frontmatter, re.MULTILINE)
                if inline_match:
                    tags = inline_match.group(1).split(',')
                    return [t.strip().strip('"\'') for t in tags]

            except Exception:
                continue

    return None


def detect_embedding_source() -> str:
    """Detect which embedding source will be used."""
    local_endpoint = os.getenv("LOCAL_EMBEDDING_ENDPOINT", "http://localhost:8080/embed")
    try:
        import urllib.request
        req = urllib.request.Request(f"{local_endpoint.rsplit('/embed', 1)[0]}/info", method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            info = json.loads(response.read().decode('utf-8'))
            return info.get('model_id', 'local-unknown')
    except:
        pass

    if os.getenv("OPENAI_API_KEY"):
        return "openai/text-embedding-3-small"
    return "unknown"


def index_memories(force: bool = False) -> dict:
    """
    Index all memories by generating embeddings.

    Args:
        force: If True, re-index all memories. Otherwise, only index new ones.

    Returns:
        Summary of indexing results.
    """
    model_source = detect_embedding_source()
    data = load_embeddings() if not force else {"memories": {}, "model": model_source}
    existing = set(data["memories"].keys())

    stats = {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}

    # Collect all memory files
    memory_files = []
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if directory.exists():
            memory_files.extend(directory.glob("*.md"))

    stats["total"] = len(memory_files)

    for path in memory_files:
        memory_id, content = parse_memory_file(path)
        if not memory_id or not content:
            stats["failed"] += 1
            continue

        # Skip if already indexed (unless forcing)
        if memory_id in existing and not force:
            stats["skipped"] += 1
            continue

        # Generate embedding
        embedding = get_embedding(content)
        if embedding:
            data["memories"][memory_id] = {
                "embedding": embedding,
                "path": str(path),
                "preview": content[:200]
            }
            stats["indexed"] += 1
            print(f"  Indexed: {memory_id}")
        else:
            stats["failed"] += 1
            print(f"  Failed: {memory_id}")

    # Update model info
    data["model"] = model_source
    data["embedding_dim"] = len(next(iter(data["memories"].values()), {}).get("embedding", [])) if data["memories"] else 0

    save_embeddings(data)
    return stats


# === v4.4: BRIDGE HIT RATE TRACKING ===
BRIDGE_LOG_FILE = MEMORY_DIR / ".bridge_hit_log.json"


def _log_bridge_hit_rate(original_query: str, bridge_terms: set, results: list):
    """
    Track whether synonym bridge terms contributed to search results.

    For each result, check if any bridge terms appear in the result's preview
    or tags but the original query terms don't fully explain the match.
    This tells us which bridges are earning their keep vs adding noise.
    """
    original_words = set(original_query.lower().split())
    bridge_hits = 0

    for r in results:
        preview = r.get("preview", "").lower()
        # Check if any bridge term appears in the result content
        for term in bridge_terms:
            if term in preview:
                bridge_hits += 1
                break

    entry = {
        "query": original_query,
        "bridge_terms": sorted(bridge_terms),
        "total_results": len(results),
        "bridge_hits": bridge_hits,
        "hit_rate": bridge_hits / len(results) if results else 0,
        "timestamp": __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat()
    }

    # Append to log file
    log = []
    if BRIDGE_LOG_FILE.exists():
        try:
            log = json.loads(BRIDGE_LOG_FILE.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            log = []
    log.append(entry)
    # Keep last 200 entries
    if len(log) > 200:
        log = log[-200:]
    BRIDGE_LOG_FILE.write_text(json.dumps(log, indent=2), encoding='utf-8')


def bridge_hit_stats() -> dict:
    """Get aggregate bridge hit rate statistics."""
    if not BRIDGE_LOG_FILE.exists():
        return {"total_queries": 0, "message": "No bridge data yet"}

    log = json.loads(BRIDGE_LOG_FILE.read_text(encoding='utf-8'))
    if not log:
        return {"total_queries": 0, "message": "No bridge data yet"}

    total = len(log)
    queries_with_bridges = sum(1 for e in log if e["bridge_terms"])
    total_hits = sum(e["bridge_hits"] for e in log)
    total_results = sum(e["total_results"] for e in log)
    avg_hit_rate = sum(e["hit_rate"] for e in log) / total if total else 0

    # Per-term tracking: which bridge terms actually hit
    term_hits = {}
    term_uses = {}
    for e in log:
        for term in e["bridge_terms"]:
            term_uses[term] = term_uses.get(term, 0) + 1
        # Can't attribute per-term hits without more granular tracking,
        # but we can track which terms are used most
    top_terms = sorted(term_uses.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_queries": total,
        "queries_with_bridges": queries_with_bridges,
        "total_bridge_hits": total_hits,
        "total_results": total_results,
        "avg_hit_rate": round(avg_hit_rate, 3),
        "aggregate_hit_rate": round(total_hits / total_results, 3) if total_results else 0,
        "top_bridge_terms": top_terms
    }


def search_memories(query: str, limit: int = 5, threshold: float = 0.3,
                    register_recall: bool = True,
                    dimension: str = None) -> list[dict]:
    """
    Search memories by semantic similarity with resolution boosting.

    When register_recall=True (default), retrieved memories are registered
    with the decay/co-occurrence system, strengthening accessed memories
    and building associative links between concepts retrieved together.

    Resolution memories (tagged with 'resolution', 'procedural', 'fix', etc.)
    get a score boost so solutions surface before problem descriptions.

    Args:
        query: Natural language query
        limit: Maximum results to return
        threshold: Minimum similarity score (0-1)
        register_recall: If True, register results as "recalled" for decay system
        dimension: v5.0 5W dimension filter ('who', 'what', 'why', 'where').
                   If set, memories with connections in this dimension get boosted.

    Returns:
        List of matching memories with scores.
    """
    data = load_embeddings()
    if not data["memories"]:
        return []

    # v4.3: Vocabulary bridging - expand query with synonyms before embedding
    # v4.4: Track bridge hit rate for vitals
    original_query = query
    bridge_terms = set()
    try:
        from synonym_bridge import expand_query
        expanded = expand_query(query)
        if expanded != query:
            # Extract which terms were added by the bridge
            original_words = set(query.lower().split())
            expanded_words = set(expanded.lower().split())
            bridge_terms = expanded_words - original_words
        query = expanded
    except Exception:
        pass  # Fail gracefully if synonym bridge unavailable

    # Get query embedding
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Failed to get query embedding", file=sys.stderr)
        return []

    # Score all memories
    results = []
    for memory_id, info in data["memories"].items():
        score = cosine_similarity(query_embedding, info["embedding"])
        if score >= threshold:
            results.append({
                "id": memory_id,
                "score": score,
                "preview": info.get("preview", "")[:150],
                "path": info.get("path", ""),
                "boosted": False
            })

    # Sort by raw score first
    results.sort(key=lambda x: x["score"], reverse=True)

    # === RESOLUTION BOOSTING (optimized) ===
    # Only check tags for top candidates (limit * 3), not all above-threshold results.
    # load_memory_tags() does a full directory scan per call, so checking all 400+
    # memories was taking ~7 seconds. Checking top ~15 takes ~0.3s.
    boost_candidates = min(limit * 3, len(results))
    for result in results[:boost_candidates]:
        tags = load_memory_tags(result["id"])
        if tags and RESOLUTION_TAGS.intersection(set(t.lower() for t in tags)):
            result["score"] *= RESOLUTION_BOOST
            result["boosted"] = True

    # Re-sort after boosting (only top candidates may have changed)
    results[:boost_candidates] = sorted(results[:boost_candidates],
                                         key=lambda x: x["score"], reverse=True)

    # === v5.0: 5W DIMENSIONAL BOOSTING ===
    # If a dimension is specified, boost memories that have connections in that
    # dimension's context graph. This surfaces dimension-relevant memories.
    if dimension and results:
        try:
            from context_manager import load_graph
            dim_graph = load_graph(dimension)
            dim_nodes = set()
            for edge_key in dim_graph.get("edges", {}):
                parts = edge_key.split("|")
                dim_nodes.update(parts)

            dim_boost_count = min(limit * 3, len(results))
            for result in results[:dim_boost_count]:
                if result["id"] in dim_nodes:
                    result["score"] *= 1.15  # 15% boost for dimension membership
                    result["dim_boosted"] = True

            results[:dim_boost_count] = sorted(
                results[:dim_boost_count],
                key=lambda x: x["score"], reverse=True
            )
        except Exception:
            pass  # Fail gracefully if context_manager unavailable

    top_results = results[:limit]

    # v4.3: Epsilon-greedy injection - occasionally surface forgotten memories
    if register_recall:
        try:
            from memory_excavation import epsilon_inject
            top_results = epsilon_inject(top_results)
        except Exception:
            pass  # Fail gracefully if excavation module unavailable

    # Register recalls with the memory system (strengthens memories + builds co-occurrence)
    if register_recall and top_results:
        try:
            from memory_manager import recall_memory
            for r in top_results:
                # This updates recall_count and adds to session tracking for co-occurrence
                recall_memory(r["id"])
        except Exception:
            pass  # Fail gracefully if memory_manager unavailable

    # v4.4: Log bridge hit rate for vitals tracking
    if bridge_terms and top_results:
        try:
            _log_bridge_hit_rate(original_query, bridge_terms, top_results)
        except Exception:
            pass  # Non-critical, never block search

    return top_results


def get_status() -> dict:
    """Get status of the semantic search index."""
    data = load_embeddings()

    # Count actual memory files
    memory_count = 0
    for directory in [CORE_DIR, ACTIVE_DIR]:
        if directory.exists():
            memory_count += len(list(directory.glob("*.md")))

    indexed_count = len(data.get("memories", {}))

    return {
        "indexed": indexed_count,
        "total_memories": memory_count,
        "coverage": f"{indexed_count}/{memory_count}",
        "model": data.get("model", "unknown"),
        "index_file": str(EMBEDDINGS_FILE),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "local_endpoint": os.getenv("LOCAL_EMBEDDING_ENDPOINT", "not configured")
    }


def embed_single(memory_id: str, content: str) -> bool:
    """
    Embed a single memory (call this when storing new memories).
    Returns True if successful.
    """
    embedding = get_embedding(content)
    if not embedding:
        return False

    data = load_embeddings()
    data["memories"][memory_id] = {
        "embedding": embedding,
        "preview": content[:200]
    }
    save_embeddings(data)
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search for SpindriftMend's memories")
    parser.add_argument("command", choices=["index", "search", "status", "bridge-stats"],
                       help="Command to run")
    parser.add_argument("query", nargs="?", help="Search query (for search command)")
    parser.add_argument("--limit", type=int, default=5, help="Max results")
    parser.add_argument("--force", action="store_true", help="Force re-index all")
    parser.add_argument("--threshold", type=float, default=0.3, help="Min similarity")
    parser.add_argument("--no-recall", action="store_true",
                       help="Skip recall registration (for thought priming - avoids co-occurrence processing)")
    parser.add_argument("--dimension", choices=["who", "what", "why", "where"],
                       help="v5.0: Boost results connected in this 5W dimension")

    args = parser.parse_args()

    if args.command == "status":
        status = get_status()
        print("Semantic Search Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    elif args.command == "index":
        print("Indexing memories...")
        stats = index_memories(force=args.force)
        print(f"\nResults:")
        print(f"  Indexed: {stats['indexed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total: {stats['total']}")

    elif args.command == "search":
        if not args.query:
            print("Error: search requires a query")
            sys.exit(1)

        results = search_memories(args.query, limit=args.limit, threshold=args.threshold,
                                  register_recall=not args.no_recall,
                                  dimension=args.dimension)

        if not results:
            print("No matching memories found.")
        else:
            print(f"Found {len(results)} matching memories:\n")
            for r in results:
                boost_marker = " [RESOLUTION]" if r.get('boosted') else ""
                dim_marker = f" [5W:{args.dimension}]" if r.get('dim_boosted') else ""
                print(f"[{r['score']:.3f}]{boost_marker}{dim_marker} {r['id']}")
                print(f"  {r['preview']}...")
                print()

    elif args.command == "bridge-stats":
        stats = bridge_hit_stats()
        print("Vocabulary Bridge Hit Rate Stats:")
        print(f"  Total queries tracked: {stats['total_queries']}")
        print(f"  Queries with bridges: {stats.get('queries_with_bridges', 0)}")
        print(f"  Total bridge hits: {stats.get('total_bridge_hits', 0)}")
        print(f"  Avg hit rate per query: {stats.get('avg_hit_rate', 0):.1%}")
        print(f"  Aggregate hit rate: {stats.get('aggregate_hit_rate', 0):.1%}")
        top = stats.get('top_bridge_terms', [])
        if top:
            print(f"  Top bridge terms:")
            for term, count in top:
                print(f"    {term}: used {count}x")
