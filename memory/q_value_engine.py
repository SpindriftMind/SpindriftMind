#!/usr/bin/env python3
"""
Q-Value Learning Engine for Memory Retrieval (MemRL-inspired)

Gives each memory a learned utility score that converges toward its true
expected retrieval value. Memories that consistently help produce good
outputs get higher Q-values; memories that get recalled but never
contribute get lower Q-values.

Based on: MemRL (arXiv:2601.03192), adapted for our memory system.

Core update rule: Q <- Q + α(r - Q)
Composite score:  λ × similarity + (1-λ) × Q-value

Usage:
    python q_value_engine.py stats          # Q-value distribution
    python q_value_engine.py top [N]        # Top N by Q-value
    python q_value_engine.py bottom [N]     # Bottom N by Q-value
    python q_value_engine.py history <id>   # Q trajectory for memory
    python q_value_engine.py convergence    # Convergence report
"""

import sys
from pathlib import Path

from memory_common import get_db

# ---------------------------------------------------------------------------
# Core Parameters (from MemRL paper, tuned for our scale)
# ---------------------------------------------------------------------------

ALPHA = 0.1              # Learning rate
LAMBDA = 0.5             # Similarity-utility balance (0.5 = equal weight)
DEFAULT_Q = 0.5           # Optimistic initialization
Q_MIN, Q_MAX = 0.0, 1.0  # Clamp bounds

# Feature flags (gradual rollout)
Q_RERANKING_ENABLED = True
Q_UPDATES_ENABLED = True

# Reward signals
REWARD_RE_RECALL = 1.0        # Recalled by 2+ different sources in session
REWARD_DOWNSTREAM = 0.8       # Led to new memory creation (appears in caused_by)
REWARD_EXPLICIT_POS = 1.0     # Manual "productive" log
REWARD_DEAD_END = -0.3        # Recalled but unused (no evidence of utility)

# Reward weights for composite
WEIGHT_RE_RECALL = 0.4
WEIGHT_DOWNSTREAM = 0.4
WEIGHT_EXPLICIT = 0.2


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def update_q(old_q: float, reward: float) -> float:
    """Q <- Q + α(r - Q). Converges toward the reward signal."""
    return clamp(old_q + ALPHA * (reward - old_q), Q_MIN, Q_MAX)


def composite_score(similarity: float, q_value: float) -> float:
    """Two-phase retrieval: λ×sim + (1-λ)×Q."""
    return LAMBDA * similarity + (1 - LAMBDA) * q_value


def compute_reward(memory_id: str, session_recalls: dict,
                   created_this_session: set) -> tuple[float, str]:
    """
    Compute composite reward from session evidence.

    Args:
        memory_id: The memory to score
        session_recalls: {source: [memory_ids]} from session_state
        created_this_session: Set of memory IDs created this session

    Returns:
        (reward, source_label) tuple
    """
    signals = []

    # Signal 1: Re-recall — recalled by 2+ different sources
    sources_that_recalled = 0
    for source, ids in session_recalls.items():
        if memory_id in ids:
            sources_that_recalled += 1
    if sources_that_recalled >= 2:
        signals.append((REWARD_RE_RECALL, WEIGHT_RE_RECALL, "re_recall"))

    # Signal 2: Downstream — this memory appears in caused_by of a new memory
    db = get_db()
    if db and created_this_session:
        for new_id in created_this_session:
            new_mem = db.get_memory(new_id)
            if new_mem:
                extra = new_mem.get('extra_metadata', {}) or {}
                caused_by = extra.get('caused_by', [])
                if memory_id in caused_by:
                    signals.append((REWARD_DOWNSTREAM, WEIGHT_DOWNSTREAM, "downstream"))
                    break  # One downstream hit is enough

    # Compute weighted reward
    if signals:
        total_weight = sum(w for _, w, _ in signals)
        weighted_reward = sum(r * w for r, w, _ in signals) / total_weight
        source = "+".join(s for _, _, s in signals)
        return (weighted_reward, source)

    # No evidence of utility — dead end
    return (REWARD_DEAD_END, "dead_end")


# ---------------------------------------------------------------------------
# Session-End Batch Update
# ---------------------------------------------------------------------------

def session_end_q_update(session_id: int = None) -> dict:
    """
    Main entry point: compute rewards and batch-update all recalled memories.
    Called from stop.py hook at session end.

    Returns summary dict with update stats.
    """
    if not Q_UPDATES_ENABLED:
        return {"updated": 0, "skipped": True, "reason": "Q_UPDATES_ENABLED=False"}

    import session_state
    session_state.load()

    db = get_db()
    if db is None:
        return {"updated": 0, "error": "no_db"}

    # Get session data
    retrieved = session_state.get_retrieved_list()
    recalls_by_source = session_state.get_recalls_by_source()
    sid = session_id or session_state.get_session_id()

    if not retrieved:
        return {"updated": 0, "reason": "no_recalls"}

    # Get memories created this session
    created_this_session = set()
    if hasattr(session_state, 'get_created_list'):
        created_this_session = set(session_state.get_created_list())

    # Batch-fetch current Q-values
    q_values = db.get_q_values(retrieved)

    results = {
        "updated": 0,
        "total_reward": 0.0,
        "avg_reward": 0.0,
        "updates": [],
    }

    import psycopg2.extras

    with db._conn() as conn:
        with conn.cursor() as cur:
            for mem_id in retrieved:
                old_q = q_values.get(mem_id, DEFAULT_Q)
                reward, source = compute_reward(
                    mem_id, recalls_by_source, created_this_session
                )
                new_q = update_q(old_q, reward)

                # Update memory's Q-value
                cur.execute(
                    f"UPDATE {db._table('memories')} SET q_value = %s WHERE id = %s",
                    (new_q, mem_id)
                )

                # Log to history
                cur.execute(f"""
                    INSERT INTO {db._table('q_value_history')}
                    (memory_id, session_id, old_q, new_q, reward, reward_source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (mem_id, sid, old_q, new_q, reward, source))

                results["updated"] += 1
                results["total_reward"] += reward
                results["updates"].append({
                    "id": mem_id,
                    "old_q": round(old_q, 4),
                    "new_q": round(new_q, 4),
                    "reward": round(reward, 4),
                    "source": source,
                })

    if results["updated"] > 0:
        results["avg_reward"] = round(
            results["total_reward"] / results["updated"], 4
        )

    return results


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

def q_stats() -> dict:
    """Q-value distribution summary."""
    db = get_db()
    if db is None:
        return {"error": "no_db"}

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE q_value != 0.5) as trained,
                    COUNT(*) FILTER (WHERE q_value >= 0.7) as high,
                    COUNT(*) FILTER (WHERE q_value <= 0.3) as low,
                    AVG(q_value) as avg_q,
                    MIN(q_value) as min_q,
                    MAX(q_value) as max_q
                FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
            """)
            row = cur.fetchone()
            return {
                "total": row[0],
                "trained": row[1],
                "high_q": row[2],
                "low_q": row[3],
                "avg_q": round(float(row[4] or 0.5), 4),
                "min_q": round(float(row[5] or 0.0), 4),
                "max_q": round(float(row[6] or 1.0), 4),
            }


def q_top(n: int = 10) -> list:
    """Top N memories by Q-value."""
    db = get_db()
    if db is None:
        return []

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, q_value, type, recall_count,
                       LEFT(content, 100) as preview
                FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
                ORDER BY q_value DESC
                LIMIT %s
            """, (n,))
            return [dict(r) for r in cur.fetchall()]


def q_bottom(n: int = 10) -> list:
    """Bottom N memories by Q-value."""
    db = get_db()
    if db is None:
        return []

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, q_value, type, recall_count,
                       LEFT(content, 100) as preview
                FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
                  AND q_value != 0.5
                ORDER BY q_value ASC
                LIMIT %s
            """, (n,))
            return [dict(r) for r in cur.fetchall()]


def q_history(memory_id: str) -> list:
    """Q-value trajectory for a specific memory."""
    db = get_db()
    if db is None:
        return []

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT session_id, old_q, new_q, reward, reward_source, updated_at
                FROM {db._table('q_value_history')}
                WHERE memory_id = %s
                ORDER BY updated_at ASC
            """, (memory_id,))
            return [dict(r) for r in cur.fetchall()]


def convergence_report() -> dict:
    """How well has the Q-value system converged?"""
    db = get_db()
    if db is None:
        return {"error": "no_db"}

    import psycopg2.extras
    with db._conn() as conn:
        with conn.cursor() as cur:
            # Count memories with non-default Q
            cur.execute(f"""
                SELECT COUNT(*) FROM {db._table('memories')}
                WHERE type IN ('active', 'core') AND q_value != 0.5
            """)
            trained = cur.fetchone()[0]

            cur.execute(f"""
                SELECT COUNT(*) FROM {db._table('memories')}
                WHERE type IN ('active', 'core')
            """)
            total = cur.fetchone()[0]

            # Average update magnitude (last 50 updates)
            cur.execute(f"""
                SELECT AVG(ABS(new_q - old_q))
                FROM (
                    SELECT new_q, old_q
                    FROM {db._table('q_value_history')}
                    ORDER BY updated_at DESC
                    LIMIT 50
                ) recent
            """)
            avg_delta = cur.fetchone()[0]

            # Memories with stabilized Q (variance of last 5 updates < 0.05)
            cur.execute(f"""
                SELECT memory_id, VARIANCE(new_q) as var_q
                FROM (
                    SELECT memory_id, new_q,
                           ROW_NUMBER() OVER (PARTITION BY memory_id ORDER BY updated_at DESC) as rn
                    FROM {db._table('q_value_history')}
                ) ranked
                WHERE rn <= 5
                GROUP BY memory_id
                HAVING COUNT(*) >= 3
            """)
            stable_count = sum(1 for row in cur.fetchall()
                              if row[1] is not None and row[1] < 0.05)

    return {
        "total_active_core": total,
        "trained_count": trained,
        "trained_pct": round(trained * 100 / total, 1) if total > 0 else 0,
        "avg_update_magnitude": round(float(avg_delta or 0), 4),
        "stabilized_count": stable_count,
        "converging": (avg_delta or 1) < 0.05 if avg_delta is not None else False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'stats':
        stats = q_stats()
        print("Q-Value Distribution")
        print("=" * 40)
        print(f"  Total (active+core): {stats.get('total', 0)}")
        print(f"  Trained (Q != 0.5):  {stats.get('trained', 0)}")
        print(f"  High Q (>= 0.7):    {stats.get('high_q', 0)}")
        print(f"  Low Q (<= 0.3):     {stats.get('low_q', 0)}")
        print(f"  Avg Q: {stats.get('avg_q', 0.5)}")
        print(f"  Range: [{stats.get('min_q', 0)}, {stats.get('max_q', 1)}]")

    elif cmd == 'top':
        n = int(args[1]) if len(args) > 1 else 10
        results = q_top(n)
        print(f"Top {n} memories by Q-value:")
        for r in results:
            print(f"  [Q:{r['q_value']:.3f}] {r['id']} (recalls={r['recall_count']}) {r['preview'][:60]}...")

    elif cmd == 'bottom':
        n = int(args[1]) if len(args) > 1 else 10
        results = q_bottom(n)
        if not results:
            print("No memories with non-default Q-values yet.")
        else:
            print(f"Bottom {n} memories by Q-value:")
            for r in results:
                print(f"  [Q:{r['q_value']:.3f}] {r['id']} (recalls={r['recall_count']}) {r['preview'][:60]}...")

    elif cmd == 'history':
        if len(args) < 2:
            print("Usage: python q_value_engine.py history <memory_id>")
            sys.exit(1)
        mem_id = args[1]
        history = q_history(mem_id)
        if not history:
            print(f"No Q-value history for {mem_id}")
        else:
            print(f"Q-value trajectory for {mem_id}:")
            for h in history:
                ts = str(h['updated_at'])[:19]
                print(f"  [{ts}] {h['old_q']:.3f} -> {h['new_q']:.3f} (r={h['reward']:.3f}, {h['reward_source']})")

    elif cmd == 'convergence':
        report = convergence_report()
        print("Q-Value Convergence Report")
        print("=" * 40)
        print(f"  Active+Core: {report.get('total_active_core', 0)}")
        print(f"  Trained:     {report.get('trained_count', 0)} ({report.get('trained_pct', 0)}%)")
        print(f"  Stabilized:  {report.get('stabilized_count', 0)}")
        print(f"  Avg |ΔQ|:    {report.get('avg_update_magnitude', 0)}")
        print(f"  Converging:  {'Yes' if report.get('converging') else 'Not yet'}")

    else:
        print(f"Unknown command: {cmd}")
        print("Available: stats, top, bottom, history, convergence")
        sys.exit(1)


if __name__ == '__main__':
    main()
