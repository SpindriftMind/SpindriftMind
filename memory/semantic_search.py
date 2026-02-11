#!/usr/bin/env python3
"""
Semantic Search for SpindriftMend's Memory System

Enables natural language queries like "what do I know about bounties?"
instead of requiring exact memory IDs.

v6.0: PostgreSQL + pgvector backend (migrated from embeddings.json)
- Embeddings stored in spin.text_embeddings via MemoryDB
- Similarity search via pgvector cosine distance (halfvec)
- Memory metadata (tags, entities, etc.) from JOIN with memories table

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

# Embedding dimensions vary by model:
# - OpenAI text-embedding-3-small: 1536
# - Qwen3-Embedding-8B: 4096
# We don't enforce dimension - just compare what we have

# === RESOLUTION BOOSTING (v2.19 from DriftCornwall) ===
# Memories tagged as resolution/fix/solution get score boost so solutions
# surface before problem descriptions when searching.
RESOLUTION_TAGS = {'resolution', 'procedural', 'fix', 'solution', 'howto', 'api', 'endpoint'}
RESOLUTION_BOOST = 1.25  # 25% score boost for resolution memories


def _get_db():
    """Get DB instance with graceful fallback."""
    try:
        from memory_common import get_db
        return get_db()
    except Exception as e:
        print(f"Warning: DB unavailable: {e}", file=sys.stderr)
        return None


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


def parse_memory_file(path: Path) -> tuple[Optional[str], Optional[str]]:
    """Parse a memory file and return (id, content). Kept for backward compat."""
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
    """
    Load tags for a memory.
    v6.0: Uses DB lookup instead of scanning all .md files.
    """
    db = _get_db()
    if db:
        try:
            mem = db.get_memory(memory_id)
            if mem:
                tags = mem.get('tags')
                # DB returns tags as a list already (PostgreSQL array)
                if tags:
                    return list(tags)
            return None
        except Exception:
            pass

    # Fallback: scan files (legacy path)
    return _load_memory_tags_file(memory_id)


def _load_memory_tags_file(memory_id: str) -> Optional[list[str]]:
    """Legacy file-based tag loading. Fallback if DB unavailable."""
    import re

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

                id_match = re.search(r'^id:\s*[\'"]?([^\'"]+)[\'"]?\s*$', frontmatter, re.MULTILINE)
                if not id_match or id_match.group(1).strip() != memory_id:
                    continue

                tags_section = re.search(r'^tags:\s*\n((?:\s*-\s*.+\n?)+)', frontmatter, re.MULTILINE)
                if tags_section:
                    tags = re.findall(r'-\s*(.+)', tags_section.group(1))
                    return [t.strip().strip('"\'') for t in tags]

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
    Index all memories by generating embeddings and storing in PostgreSQL.

    v6.0: Reads memories from DB, generates embeddings, stores via db.store_embedding().

    Args:
        force: If True, re-index all memories. Otherwise, only index new ones.

    Returns:
        Summary of indexing results.
    """
    db = _get_db()
    if not db:
        print("Error: DB unavailable, cannot index memories.", file=sys.stderr)
        return {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}

    model_source = detect_embedding_source()
    stats = {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}

    # Get all memories from DB (active + core)
    memories = []
    for type_ in ['active', 'core']:
        memories.extend(db.list_memories(type_=type_, limit=10000))
    stats["total"] = len(memories)

    # If not forcing, get set of already-indexed memory IDs
    existing_ids = set()
    if not force:
        try:
            with db._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT memory_id FROM {db._table('text_embeddings')}")
                    existing_ids = {row[0] for row in cur.fetchall()}
        except Exception as e:
            print(f"Warning: Could not check existing embeddings: {e}", file=sys.stderr)

    for mem in memories:
        memory_id = mem['id']
        content = mem.get('content', '')

        if not memory_id or not content:
            stats["failed"] += 1
            continue

        # Skip if already indexed (unless forcing)
        if memory_id in existing_ids and not force:
            stats["skipped"] += 1
            continue

        # Generate embedding
        embedding = get_embedding(content)
        if embedding:
            try:
                db.store_embedding(memory_id, embedding,
                                   preview=content[:200],
                                   model=model_source)
                stats["indexed"] += 1
                print(f"  Indexed: {memory_id}")
            except Exception as e:
                stats["failed"] += 1
                print(f"  Failed (DB): {memory_id}: {e}")
        else:
            stats["failed"] += 1
            print(f"  Failed (embedding): {memory_id}")

    return stats


# === v4.4: BRIDGE HIT RATE TRACKING ===
BRIDGE_LOG_FILE = MEMORY_DIR / ".bridge_hit_log.json"


def _log_bridge_hit_rate(original_query: str, bridge_terms: set, results: list):
    """
    Track whether synonym bridge terms contributed to search results.
    v4.5: Per-term hit tracking for zero-hit bridge detection.

    For each result, check if any bridge terms appear in the result's preview.
    Tracks which specific terms matched (not just aggregate count).
    """
    original_words = set(original_query.lower().split())
    bridge_hits = 0
    # v4.5: Track which specific terms hit in this query
    terms_that_hit = set()

    for r in results:
        preview = r.get("preview", "").lower()
        result_hit = False
        for term in bridge_terms:
            if term in preview:
                terms_that_hit.add(term)
                if not result_hit:
                    bridge_hits += 1
                    result_hit = True

    # v4.6: Classify query domain for per-domain bridge analysis
    query_domains = []
    try:
        from topic_context import classify_content
        query_domains = classify_content(original_query)
    except Exception:
        pass

    entry = {
        "query": original_query,
        "query_domains": query_domains,
        "bridge_terms": sorted(bridge_terms),
        "terms_that_hit": sorted(terms_that_hit),
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
    """Get aggregate bridge hit rate statistics with per-term analysis."""
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

    # v4.5: Per-term hit tracking (uses terms_that_hit field if available)
    term_uses = {}  # How many queries each term was expanded into
    term_hits = {}  # How many queries each term actually contributed to results
    for e in log:
        for term in e.get("bridge_terms", []):
            term_uses[term] = term_uses.get(term, 0) + 1
        for term in e.get("terms_that_hit", []):
            term_hits[term] = term_hits.get(term, 0) + 1

    # Top performing terms (sorted by hit count)
    top_terms = sorted(term_uses.items(), key=lambda x: term_hits.get(x[0], 0), reverse=True)[:10]
    top_terms_with_hits = [
        (term, uses, term_hits.get(term, 0))
        for term, uses in top_terms
    ]

    # v4.5: Zero-hit bridge detection
    zero_hit_terms = [
        (term, uses) for term, uses in term_uses.items()
        if term_hits.get(term, 0) == 0
    ]
    zero_hit_terms.sort(key=lambda x: x[1], reverse=True)

    # v4.6: Per-domain bridge effectiveness
    domain_stats = {}
    for e in log:
        for domain in e.get("query_domains", []):
            if domain not in domain_stats:
                domain_stats[domain] = {"queries": 0, "hits": 0, "results": 0}
            domain_stats[domain]["queries"] += 1
            domain_stats[domain]["hits"] += e.get("bridge_hits", 0)
            domain_stats[domain]["results"] += e.get("total_results", 0)

    return {
        "total_queries": total,
        "queries_with_bridges": queries_with_bridges,
        "total_bridge_hits": total_hits,
        "total_results": total_results,
        "avg_hit_rate": round(avg_hit_rate, 3),
        "aggregate_hit_rate": round(total_hits / total_results, 3) if total_results else 0,
        "top_bridge_terms": top_terms_with_hits,
        "zero_hit_terms": zero_hit_terms[:20],
        "zero_hit_count": len(zero_hit_terms),
        "total_unique_terms": len(term_uses),
        "domain_stats": domain_stats,
    }


def bridge_zero_hits(min_queries: int = 10) -> list:
    """
    Flag bridge terms with 0 hits after sufficient exposure.
    These are candidates for review -- either the synonym mapping is wrong
    or the term is domain-specific and just hasn't been queried yet.

    Args:
        min_queries: Minimum queries a term must appear in before flagging (default 10)
    """
    if not BRIDGE_LOG_FILE.exists():
        return []

    log = json.loads(BRIDGE_LOG_FILE.read_text(encoding='utf-8'))
    term_uses = {}
    term_hits = {}
    for e in log:
        for term in e.get("bridge_terms", []):
            term_uses[term] = term_uses.get(term, 0) + 1
        for term in e.get("terms_that_hit", []):
            term_hits[term] = term_hits.get(term, 0) + 1

    flagged = [
        {"term": term, "queries_seen": uses, "hits": 0}
        for term, uses in term_uses.items()
        if uses >= min_queries and term_hits.get(term, 0) == 0
    ]
    flagged.sort(key=lambda x: x["queries_seen"], reverse=True)
    return flagged


def search_memories(query: str, limit: int = 5, threshold: float = 0.3,
                    register_recall: bool = True,
                    dimension: str = None) -> list[dict]:
    """
    Search memories by semantic similarity with resolution boosting.

    v6.0: Uses pgvector for similarity search instead of in-Python cosine.
    The DB does the heavy lifting (cosine distance on halfvec), then we
    apply resolution boosting, 5W dimensional boosting, and epsilon injection
    as post-processing in Python.

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
    db = _get_db()
    if not db:
        print("Warning: DB unavailable, returning empty results.", file=sys.stderr)
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

    # === pgvector SEARCH ===
    # Fetch more results than needed to allow for post-processing filtering/boosting
    fetch_limit = max(limit * 3, 20)
    try:
        raw_results = db.search_embeddings(query_embedding, limit=fetch_limit)
    except Exception as e:
        print(f"DB search error: {e}", file=sys.stderr)
        return []

    # Transform DB results into the expected format and apply threshold
    results = []
    for row in raw_results:
        similarity = float(row.get('similarity', 0))
        if similarity < threshold:
            continue
        results.append({
            "id": row['id'],
            "score": similarity,
            "preview": (row.get('preview') or row.get('content', ''))[:150],
            "boosted": False,
            # Carry forward DB data for resolution boosting (avoid extra queries)
            "_tags": row.get('tags'),
        })

    # Sort by raw score first (should already be sorted from DB, but be safe)
    results.sort(key=lambda x: x["score"], reverse=True)

    # === ENTITY INDEX INJECTION (WHO dimension fix) ===
    # When query mentions a known contact, inject their memories into candidates.
    # This bridges the gap between contact names and memory embeddings.
    # Credit: DriftCornwall's entity_index.py pattern
    try:
        from entity_index import get_memories_for_query
        entity_mem_ids = get_memories_for_query(original_query)
        if entity_mem_ids:
            existing_ids = {r["id"] for r in results}
            for mem_id in entity_mem_ids[:10]:  # Cap at 10 injected
                if mem_id not in existing_ids:
                    # Load preview from DB
                    preview = ""
                    mem_score = threshold
                    try:
                        mem_row = db.get_memory(mem_id)
                        if mem_row:
                            preview = (mem_row.get('content') or '')[:150]
                    except Exception:
                        pass

                    # Strong boost for entity-matched memories (2x baseline)
                    boosted_score = max(mem_score * 2.0, threshold + 0.3)
                    results.append({
                        "id": mem_id,
                        "score": boosted_score,
                        "preview": preview,
                        "boosted": False,
                        "_tags": None,
                        "entity_injected": True,
                    })
                else:
                    # Boost existing results that match entity (1.8x)
                    for r in results:
                        if r["id"] == mem_id:
                            r["score"] *= 1.8
                            r["entity_match"] = True
                            break
    except ImportError:
        pass
    except Exception:
        pass

    # === GRAVITY WELL DAMPENING ===
    # Penalize memories where query key terms don't appear in the preview.
    # This catches hub memories that match everything via embedding similarity
    # but don't actually contain the relevant information.
    # Credit: DriftCornwall's gravity well dampening pattern
    query_terms = set(original_query.lower().split())
    stopwords = {'what', 'is', 'my', 'the', 'a', 'an', 'do', 'i', 'know', 'about',
                 'have', 'did', 'on', 'in', 'for', 'to', 'of', 'and', 'or', 'how',
                 'why', 'where', 'when', 'who', 'should', 'today', 'done', 'been',
                 'can', 'will', 'would', 'could', 'does', 'has', 'was', 'are', 'with'}
    key_terms = query_terms - stopwords
    if key_terms and len(key_terms) >= 1:
        for result in results:
            # Skip dampening for entity-matched memories — we KNOW they're relevant
            if result.get("entity_injected") or result.get("entity_match"):
                continue
            preview_lower = result.get("preview", "").lower()
            term_overlap = sum(1 for t in key_terms if t in preview_lower)
            if term_overlap == 0 and result["score"] > threshold:
                # No key terms found in preview — dampen by 50%
                result["score"] *= 0.5
                result["dampened"] = True

    # === HUB DEGREE DAMPENING ===
    # Memories with very high co-occurrence degree are connected to everything.
    # They score well on embedding similarity for any query but rarely contain
    # the specific information sought. Penalize based on degree above median.
    # This complements keyword dampening (catches hubs whose preview happens
    # to contain query terms) and entity exemption still applies.
    try:
        _hub_result_ids = [r["id"] for r in results if not r.get("entity_injected") and not r.get("entity_match")]
        if _hub_result_ids:
            _hub_edges = db.get_all_edges()
            _hub_degrees = {}
            for pair_key in _hub_edges:
                ids = pair_key.split('|')
                if len(ids) == 2:
                    for mid in ids:
                        _hub_degrees[mid] = _hub_degrees.get(mid, 0) + 1
            if _hub_degrees:
                _all_degrees = sorted(_hub_degrees.values())
                _median_degree = _all_degrees[len(_all_degrees) // 2] if _all_degrees else 1
                _p90_degree = _all_degrees[int(len(_all_degrees) * 0.9)] if _all_degrees else 1
                # Only dampen memories above P90 degree (top 10% hubs)
                for result in results:
                    if result.get("entity_injected") or result.get("entity_match"):
                        continue
                    mem_degree = _hub_degrees.get(result["id"], 0)
                    if mem_degree > _p90_degree:
                        # Scale: at P90 = 1.0x (no dampening), at max = 0.6x
                        # Formula: 1.0 - 0.4 * (degree - P90) / (max - P90)
                        _max_degree = _all_degrees[-1] if _all_degrees else _p90_degree + 1
                        if _max_degree > _p90_degree:
                            hub_factor = 1.0 - 0.4 * (mem_degree - _p90_degree) / (_max_degree - _p90_degree)
                            hub_factor = max(hub_factor, 0.6)  # Floor at 60%
                            result["score"] *= hub_factor
                            result["hub_dampened"] = True
                            result["hub_degree"] = mem_degree
    except Exception:
        pass  # Non-critical — fail gracefully

    # === RESOLUTION BOOSTING ===
    # v6.0: Tags come from the JOIN result, no extra file scanning needed
    boost_candidates = min(limit * 3, len(results))
    for result in results[:boost_candidates]:
        tags = result.get("_tags")
        if tags and RESOLUTION_TAGS.intersection(set(t.lower() for t in tags)):
            result["score"] *= RESOLUTION_BOOST
            result["boosted"] = True

    # Re-sort after boosting (only top candidates may have changed)
    results[:boost_candidates] = sorted(results[:boost_candidates],
                                         key=lambda x: x["score"], reverse=True)

    # === v5.1: 5W DIMENSIONAL BOOSTING (auto-detect + log-degree) ===
    # Auto-detect dimension from session context if not specified.
    # Boost proportional to log(1 + degree) in dimensional graph.
    active_dimension = dimension
    if not active_dimension and results:
        try:
            from context_manager import get_session_dimensions
            dims = get_session_dimensions()
            if dims:
                active_dimension = dims[0]  # Highest-scoring dimension
        except Exception:
            pass

    if active_dimension and results:
        try:
            from math import log
            from memory_common import get_db as _get_search_db
            search_db = _get_search_db()
            if search_db is not None:
                # DB-first degree lookup — no need to load full graph
                candidate_ids = [r["id"] for r in results[:min(limit * 3, len(results))]]
                degrees = search_db.get_dimension_degree(
                    active_dimension, "", candidate_ids
                )
                dim_boost_count = min(limit * 3, len(results))
                for result in results[:dim_boost_count]:
                    degree = degrees.get(result["id"], 0)
                    if degree > 0:
                        # Logarithmic boost: grows slowly, prevents hub domination
                        result["score"] *= (1 + 0.1 * log(1 + degree))
                        result["dim_boosted"] = True
                        result["dim_dimension"] = active_dimension
                        result["dim_degree"] = degree

                results[:dim_boost_count] = sorted(
                    results[:dim_boost_count],
                    key=lambda x: x["score"], reverse=True
                )
        except Exception:
            pass  # Fail gracefully if context_manager or DB unavailable

    # Final sort of ALL results (entity-injected items may be appended past
    # the partial-sort ranges above, so we need a full re-sort)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Clean up internal fields before returning
    for result in results:
        result.pop("_tags", None)

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
                recall_memory(r["id"], source='search')
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
    """
    Get status of the semantic search index.
    v6.0: Uses DB stats instead of counting files and JSON entries.
    """
    db = _get_db()

    if db:
        try:
            stats = db.comprehensive_stats()
            total_memories = stats.get('total_memories', 0)
            indexed_count = stats.get('text_embeddings', 0)

            return {
                "indexed": indexed_count,
                "total_memories": total_memories,
                "coverage": f"{indexed_count}/{total_memories}",
                "backend": "postgresql+pgvector",
                "model": detect_embedding_source(),
                "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
                "local_endpoint": os.getenv("LOCAL_EMBEDDING_ENDPOINT", "not configured")
            }
        except Exception as e:
            print(f"Warning: DB stats failed: {e}", file=sys.stderr)

    # Fallback: report what we can without DB
    return {
        "indexed": 0,
        "total_memories": 0,
        "coverage": "unknown (DB unavailable)",
        "backend": "postgresql+pgvector (OFFLINE)",
        "model": detect_embedding_source(),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "local_endpoint": os.getenv("LOCAL_EMBEDDING_ENDPOINT", "not configured")
    }


def embed_single(memory_id: str, content: str) -> bool:
    """
    Embed a single memory (call this when storing new memories).
    v6.0: Stores embedding in PostgreSQL via db.store_embedding().
    Returns True if successful.
    """
    embedding = get_embedding(content)
    if not embedding:
        return False

    db = _get_db()
    if not db:
        print(f"Warning: DB unavailable, could not store embedding for {memory_id}",
              file=sys.stderr)
        return False

    try:
        model = detect_embedding_source()
        db.store_embedding(memory_id, embedding,
                           preview=content[:200],
                           model=model)
        return True
    except Exception as e:
        print(f"Error storing embedding for {memory_id}: {e}", file=sys.stderr)
        return False


def remove_from_index(memory_id: str) -> bool:
    """
    Remove a memory's embedding from the index.
    v6.0: Deletes from spin.text_embeddings via raw SQL.
    Returns True if a row was deleted.
    """
    db = _get_db()
    if not db:
        print(f"Warning: DB unavailable, could not remove embedding for {memory_id}",
              file=sys.stderr)
        return False

    try:
        with db._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {db._table('text_embeddings')} WHERE memory_id = %s",
                    (memory_id,)
                )
                return cur.rowcount > 0
    except Exception as e:
        print(f"Error removing embedding for {memory_id}: {e}", file=sys.stderr)
        return False


def find_similar_pairs(threshold: float = 0.85, limit: int = 10) -> list[dict]:
    """
    Find pairs of memories with high similarity (consolidation candidates).
    v6.0: Uses pgvector to find similar pairs by checking each memory's
    nearest neighbors.

    Args:
        threshold: Minimum similarity score (0-1) to consider a pair
        limit: Maximum number of pairs to return

    Returns:
        List of dicts with {id1, id2, similarity, preview1, preview2}
    """
    db = _get_db()
    if not db:
        print("Warning: DB unavailable for similarity search.", file=sys.stderr)
        return []

    try:
        # Get all embeddings with their memory IDs
        pairs = []
        seen = set()

        with db._conn() as conn:
            with conn.cursor() as cur:
                # Get all memory IDs that have embeddings
                cur.execute(f"SELECT memory_id FROM {db._table('text_embeddings')}")
                all_ids = [row[0] for row in cur.fetchall()]

            # For each memory, find its nearest neighbor via pgvector
            # This is O(n) queries but each is fast with the index
            for memory_id in all_ids:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT e2.memory_id, e2.preview,
                               1 - (e1.embedding <=> e2.embedding) AS similarity
                        FROM {db._table('text_embeddings')} e1
                        JOIN {db._table('text_embeddings')} e2
                            ON e1.memory_id != e2.memory_id
                        WHERE e1.memory_id = %s
                        ORDER BY e1.embedding <=> e2.embedding
                        LIMIT 3
                    """, (memory_id,))

                    for row in cur.fetchall():
                        other_id, other_preview, sim = row
                        sim = float(sim)
                        if sim >= threshold:
                            pair_key = tuple(sorted([memory_id, other_id]))
                            if pair_key not in seen:
                                seen.add(pair_key)
                                pairs.append({
                                    "id1": pair_key[0],
                                    "id2": pair_key[1],
                                    "similarity": sim,
                                    "preview1": "",  # Can be fetched if needed
                                    "preview2": other_preview or "",
                                })

                if len(pairs) >= limit * 2:
                    break

        # Sort by similarity descending, return top results
        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        return pairs[:limit]

    except Exception as e:
        print(f"Error finding similar pairs: {e}", file=sys.stderr)
        return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search for SpindriftMend's memories")
    parser.add_argument("command", choices=["index", "search", "status", "bridge-stats", "bridge-zero-hits"],
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
                dim_label = r.get('dim_dimension') or args.dimension or '?'
                dim_marker = f" [5W:{dim_label}]" if r.get('dim_boosted') else ""
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
        print(f"  Unique bridge terms: {stats.get('total_unique_terms', 0)}")
        print(f"  Zero-hit terms: {stats.get('zero_hit_count', 0)}")
        top = stats.get('top_bridge_terms', [])
        if top:
            print(f"  Top bridge terms (by hit count):")
            for term, uses, hits in top:
                rate = f"{hits/uses:.0%}" if uses else "0%"
                print(f"    {term}: {hits}/{uses} hits ({rate})")
        domains = stats.get('domain_stats', {})
        if domains:
            print(f"  Bridge effectiveness by domain:")
            for domain, ds in sorted(domains.items(), key=lambda x: x[1]['queries'], reverse=True):
                rate = ds['hits'] / ds['results'] if ds['results'] else 0
                print(f"    {domain}: {ds['queries']} queries, {rate:.0%} hit rate")

    elif args.command == "bridge-zero-hits":
        flagged = bridge_zero_hits(min_queries=10)
        if not flagged:
            print("No zero-hit bridge terms found (with 10+ query exposure).")
        else:
            print(f"Zero-hit bridge terms ({len(flagged)} terms with 10+ queries, 0 hits):")
            print(f"These may be bad synonyms or just underexposed.\n")
            for item in flagged:
                print(f"  {item['term']}: seen in {item['queries_seen']} queries, 0 hits")
