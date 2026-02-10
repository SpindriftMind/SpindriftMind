#!/usr/bin/env python3
"""
System Vitals Monitor v1.0

Tracks all moving parts of the memory system over time.
Records a snapshot after each session, shows trends, and flags
anything that has stalled or is regressing.

Usage:
    python system_vitals.py record          # Take a snapshot now
    python system_vitals.py latest          # Show most recent snapshot
    python system_vitals.py trends [N]      # Show trends over last N sessions (default 10)
    python system_vitals.py alerts          # Flag stalled or regressing metrics
    python system_vitals.py history [N]     # Show last N snapshots raw (default 5)

Designed to be called from stop.py hook automatically.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MEMORY_ROOT = Path(__file__).parent
VITALS_LOG = MEMORY_ROOT / ".vitals_log.json"

# Alert thresholds: how many sessions of no change before we flag
STALL_THRESHOLD = 5
DECLINE_THRESHOLD = 3


def _count_files(directory, pattern="*.md"):
    """Count files matching pattern in directory."""
    d = MEMORY_ROOT / directory
    return len(list(d.glob(pattern))) if d.exists() else 0


def _load_json(path, default=None):
    """Safely load a JSON file."""
    fp = MEMORY_ROOT / path if not Path(path).is_absolute() else Path(path)
    if not fp.exists():
        return default
    try:
        return json.loads(fp.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return default


def collect_vitals():
    """Collect all system vitals from data files. Returns a snapshot dict."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {}
    }
    m = snapshot["metrics"]

    # --- MEMORY COUNTS ---
    core = _count_files("core")
    active = _count_files("active")
    archive = _count_files("archive")
    m["memory_total"] = core + active + archive
    m["memory_core"] = core
    m["memory_active"] = active
    m["memory_archive"] = archive

    # --- CO-OCCURRENCE (v3.0 edges format) ---
    edges_data = _load_json(".edges_v3.json", {})
    if isinstance(edges_data, dict) and edges_data:
        beliefs = [e.get('belief', 0) for e in edges_data.values() if isinstance(e, dict)]
        m["cooccurrence_pairs"] = sum(1 for b in beliefs if b > 0)
        m["cooccurrence_total_strength"] = round(sum(beliefs), 2)
        m["cooccurrence_links"] = sum(1 for b in beliefs if b >= 3.0)
        m["cooccurrence_edges_total"] = len(edges_data)
    else:
        m["cooccurrence_pairs"] = 0
        m["cooccurrence_total_strength"] = 0
        m["cooccurrence_links"] = 0
        m["cooccurrence_edges_total"] = 0

    # --- REJECTION LOG ---
    rej_data = _load_json(".rejection_log.json", {})
    if isinstance(rej_data, dict):
        rejections = rej_data.get('rejections', [])
    elif isinstance(rej_data, list):
        rejections = rej_data
    else:
        rejections = []
    m["rejection_count"] = len(rejections)

    # --- LESSONS ---
    lessons = _load_json("lessons.json", [])
    if isinstance(lessons, list):
        m["lesson_count"] = len(lessons)
    else:
        m["lesson_count"] = 0

    # --- MERKLE CHAIN ---
    attestations = _load_json("attestations.json", [])
    if isinstance(attestations, list) and attestations:
        latest_att = attestations[-1]
        m["merkle_chain_depth"] = latest_att.get('chain_depth', len(attestations))
        m["merkle_memory_count"] = latest_att.get('memory_count', 0)
    else:
        m["merkle_chain_depth"] = 0
        m["merkle_memory_count"] = 0

    # --- COGNITIVE FINGERPRINT ---
    fp_history = _load_json(".fingerprint_history.json", [])
    if isinstance(fp_history, list) and fp_history:
        latest_fp = fp_history[-1]
        m["fingerprint_nodes"] = latest_fp.get('node_count', 0)
        m["fingerprint_edges"] = latest_fp.get('edge_count', 0)
        m["identity_drift"] = latest_fp.get('drift_score', 0.0)
    else:
        m["fingerprint_nodes"] = 0
        m["fingerprint_edges"] = 0
        m["identity_drift"] = 0.0

    # --- SESSION RECALLS ---
    session = _load_json(".session_state.json", {})
    retrieved = session.get('retrieved', [])
    m["session_recalls"] = len(retrieved) if isinstance(retrieved, list) else 0

    # --- SOCIAL ---
    replies_data = _load_json("social/my_replies.json", {})
    if isinstance(replies_data, dict):
        replies = replies_data.get('replies', {})
        m["social_replies_tracked"] = len(replies)
    else:
        m["social_replies_tracked"] = 0

    index_data = _load_json("social/social_index.json", {})
    m["social_contacts"] = index_data.get('total_contacts', 0)

    # --- PLATFORM CONTEXT ---
    try:
        tagged = 0
        total = m["memory_total"]
        for tier in ["core", "active"]:
            d = MEMORY_ROOT / tier
            if not d.exists():
                continue
            for fp in d.glob("*.md"):
                try:
                    content = fp.read_text(encoding='utf-8', errors='replace')
                    if 'platforms:' in content[:500]:
                        tagged += 1
                except Exception:
                    pass
        m["platform_tagged"] = tagged
        m["platform_tagged_pct"] = round(tagged * 100 / total, 1) if total > 0 else 0
    except Exception:
        m["platform_tagged"] = 0
        m["platform_tagged_pct"] = 0

    # --- DECAY HISTORY ---
    decay = _load_json(".decay_history.json", {"sessions": []})
    sessions = decay.get('sessions', [])
    if sessions:
        last = sessions[-1]
        m["last_decay_count"] = last.get('decayed', 0)
        m["last_prune_count"] = last.get('pruned', 0)
    else:
        m["last_decay_count"] = 0
        m["last_prune_count"] = 0
    m["decay_sessions_recorded"] = len(sessions)

    # --- VOCABULARY ---
    vocab = _load_json("vocabulary_map.json", {})
    m["vocabulary_terms"] = len(vocab) if isinstance(vocab, dict) else 0

    # --- BRIDGE HIT RATE (v4.4) ---
    bridge_log = _load_json(".bridge_hit_log.json", [])
    if bridge_log:
        m["bridge_queries"] = len(bridge_log)
        m["bridge_hit_rate"] = round(
            sum(e.get("hit_rate", 0) for e in bridge_log) / len(bridge_log), 3
        ) if bridge_log else 0
    else:
        m["bridge_queries"] = 0
        m["bridge_hit_rate"] = 0

    # --- SEARCH INDEX ---
    # embeddings.json is large (~26MB), so just check key count efficiently
    emb_file = MEMORY_ROOT / "embeddings.json"
    if emb_file.exists():
        try:
            # Read just the start to get structure, then count memories key
            import subprocess
            result = subprocess.run(
                [sys.executable, "-c",
                 "import json;d=json.load(open('embeddings.json','r',encoding='utf-8'));print(len(d.get('memories',{})))"],
                capture_output=True, text=True, timeout=10, cwd=str(MEMORY_ROOT)
            )
            m["search_indexed"] = int(result.stdout.strip()) if result.returncode == 0 else 0
        except Exception:
            m["search_indexed"] = 0
    else:
        m["search_indexed"] = 0

    return snapshot


def load_vitals_log():
    """Load the vitals log history."""
    return _load_json(str(VITALS_LOG), [])


def save_vitals_log(log):
    """Save the vitals log, keeping last 100 snapshots."""
    log = log[-100:]
    VITALS_LOG.write_text(json.dumps(log, indent=2), encoding='utf-8')


def record_vitals():
    """Collect vitals and append to log. Returns the snapshot."""
    snapshot = collect_vitals()
    log = load_vitals_log()
    log.append(snapshot)
    save_vitals_log(log)
    return snapshot


def get_trends(window=10):
    """
    Analyze trends over the last N snapshots.
    Returns list of (metric_name, direction, current, previous_avg, detail).
    """
    log = load_vitals_log()
    if len(log) < 2:
        return []

    recent = log[-window:] if len(log) >= window else log
    latest = recent[-1]["metrics"]
    trends = []

    for key in sorted(latest.keys()):
        values = [s["metrics"].get(key) for s in recent if key in s.get("metrics", {})]
        values = [v for v in values if v is not None and isinstance(v, (int, float))]
        if not values:
            continue

        current = values[-1]
        if len(values) < 2:
            trends.append((key, "new", current, None, "first measurement"))
            continue

        prev_avg = sum(values[:-1]) / len(values[:-1])
        prev_values = values[:-1]

        # Determine direction
        # Only flag "stalled" if unchanged for STALL_THRESHOLD+ consecutive sessions
        consecutive_same = 0
        for v in reversed(values):
            if v == current:
                consecutive_same += 1
            else:
                break

        if consecutive_same >= STALL_THRESHOLD and all(v == current for v in values[-STALL_THRESHOLD:]):
            direction = "stalled"
            detail = f"unchanged for {consecutive_same} sessions"
        elif current > prev_avg * 1.05:
            direction = "growing"
            detail = f"+{current - prev_avg:.1f} vs avg"
        elif current < prev_avg * 0.95:
            # Check if declining trend
            if len(prev_values) >= 2 and all(prev_values[i] >= prev_values[i + 1] for i in range(len(prev_values) - 1)):
                direction = "declining"
            else:
                direction = "below_avg"
            detail = f"{current - prev_avg:.1f} vs avg"
        else:
            direction = "stable"
            detail = f"~{prev_avg:.1f} avg"

        trends.append((key, direction, current, round(prev_avg, 2), detail))

    return trends


def check_alerts():
    """
    Check for concerning patterns. Returns list of alert dicts.
    Each alert: {metric, severity, message, values}
    """
    log = load_vitals_log()
    alerts = []

    if len(log) < 2:
        alerts.append({
            "metric": "vitals_log",
            "severity": "info",
            "message": f"Only {len(log)} snapshot(s) recorded. Need at least 2 for trend analysis.",
            "values": []
        })
        return alerts

    recent = log[-STALL_THRESHOLD:]
    metrics_to_watch = {
        # metric: (should_grow, severity_if_stalled, description)
        "rejection_count": (True, "warn", "Taste fingerprint not building"),
        "lesson_count": (True, "warn", "No new lessons being extracted"),
        "cooccurrence_links": (True, "warn", "Co-occurrence links not growing"),
        "merkle_chain_depth": (True, "error", "Merkle chain not incrementing (attestation broken?)"),
        "social_replies_tracked": (True, "info", "No new social replies tracked"),
        "memory_total": (True, "info", "Total memory count not growing"),
    }

    for metric, (should_grow, severity, desc) in metrics_to_watch.items():
        values = [s["metrics"].get(metric) for s in recent if metric in s.get("metrics", {})]
        values = [v for v in values if v is not None]

        if len(values) < 2:
            continue

        # Check for stall
        if should_grow and len(values) >= STALL_THRESHOLD:
            if all(v == values[0] for v in values):
                alerts.append({
                    "metric": metric,
                    "severity": severity,
                    "message": f"{desc} - unchanged at {values[0]} for {len(values)} sessions",
                    "values": values
                })

        # Check for decline
        if len(values) >= DECLINE_THRESHOLD:
            tail = values[-DECLINE_THRESHOLD:]
            if all(tail[i] > tail[i + 1] for i in range(len(tail) - 1)):
                alerts.append({
                    "metric": metric,
                    "severity": "warn",
                    "message": f"{metric} declining: {' -> '.join(str(v) for v in tail)}",
                    "values": tail
                })

    # Special: session recalls = 0 repeatedly
    recall_values = [s["metrics"].get("session_recalls", 0) for s in recent]
    zero_streak = sum(1 for v in reversed(recall_values) if v == 0)
    if zero_streak >= 3:
        alerts.append({
            "metric": "session_recalls",
            "severity": "warn",
            "message": f"Session recalls were 0 for {zero_streak} consecutive sessions (graph not building)",
            "values": recall_values
        })

    # Special: co-occurrence pairs declining sharply
    pair_values = [s["metrics"].get("cooccurrence_pairs", 0) for s in log[-5:]]
    pair_values = [v for v in pair_values if v > 0]
    if len(pair_values) >= 2 and pair_values[-1] < pair_values[0] * 0.8:
        alerts.append({
            "metric": "cooccurrence_pairs",
            "severity": "warn",
            "message": f"Co-occurrence pairs dropped {pair_values[0]} -> {pair_values[-1]} ({round((1 - pair_values[-1]/pair_values[0]) * 100)}% loss)",
            "values": pair_values
        })

    # Special: identity drift tiered thresholds (Drift's suggestion)
    # Natural drift is ~0.0-0.2. Biggest observed natural drift: 0.2081 (Drift, Day 7)
    drift_values = [s["metrics"].get("identity_drift", 0) for s in recent]
    drift_values = [v for v in drift_values if isinstance(v, (int, float))]
    if drift_values and drift_values[-1] > 0.15:
        drift_val = drift_values[-1]
        if drift_val > 0.5:
            severity, desc = "error", "catastrophic topology change"
        elif drift_val > 0.3:
            severity, desc = "warn", "significant identity shift"
        else:
            severity, desc = "info", "notable topology change"
        alerts.append({
            "metric": "identity_drift",
            "severity": severity,
            "message": f"Identity drift {drift_val:.3f} ({desc})",
            "values": drift_values
        })

    if not alerts:
        alerts.append({
            "metric": "all_clear",
            "severity": "ok",
            "message": "All systems nominal. No alerts.",
            "values": []
        })

    return alerts


# === CLI FORMATTERS ===

def format_snapshot(snapshot, compact=False):
    """Format a vitals snapshot for display."""
    m = snapshot["metrics"]
    ts = snapshot.get("timestamp", "?")

    if compact:
        parts = [
            f"mem={m.get('memory_total', '?')}",
            f"pairs={m.get('cooccurrence_pairs', '?')}",
            f"links={m.get('cooccurrence_links', '?')}",
            f"rej={m.get('rejection_count', '?')}",
            f"lessons={m.get('lesson_count', '?')}",
            f"merkle={m.get('merkle_chain_depth', '?')}",
            f"drift={m.get('identity_drift', '?')}",
        ]
        return f"[{ts[:19]}] {' | '.join(parts)}"

    lines = [
        f"System Vitals Snapshot — {ts}",
        "=" * 55,
        "",
        "MEMORY",
        f"  Total: {m.get('memory_total', '?')} (core={m.get('memory_core', '?')}, active={m.get('memory_active', '?')}, archive={m.get('memory_archive', '?')})",
        "",
        "CO-OCCURRENCE",
        f"  Total edges: {m.get('cooccurrence_edges_total', '?')}",
        f"  Active pairs (belief>0): {m.get('cooccurrence_pairs', '?')}",
        f"  Total strength: {m.get('cooccurrence_total_strength', '?')}",
        f"  Links (>=3.0): {m.get('cooccurrence_links', '?')}",
        "",
        "IDENTITY",
        f"  Merkle chain depth: {m.get('merkle_chain_depth', '?')} ({m.get('merkle_memory_count', '?')} memories attested)",
        f"  Fingerprint nodes: {m.get('fingerprint_nodes', '?')} | edges: {m.get('fingerprint_edges', '?')}",
        f"  Identity drift: {m.get('identity_drift', '?')}",
        f"  Rejections logged: {m.get('rejection_count', '?')}",
        "",
        "LEARNING",
        f"  Lessons extracted: {m.get('lesson_count', '?')}",
        f"  Session recalls: {m.get('session_recalls', '?')}",
        "",
        "SOCIAL",
        f"  Contacts: {m.get('social_contacts', '?')}",
        f"  Replies tracked: {m.get('social_replies_tracked', '?')}",
        "",
        "PLATFORM",
        f"  Tagged memories: {m.get('platform_tagged', '?')} ({m.get('platform_tagged_pct', '?')}%)",
        "",
        "GRAPH HEALTH",
        f"  Last decay: {m.get('last_decay_count', '?')} pairs | pruned: {m.get('last_prune_count', '?')}",
        f"  Decay sessions recorded: {m.get('decay_sessions_recorded', '?')}",
        "",
        "SEARCH & VOCABULARY",
        f"  Indexed memories: {m.get('search_indexed', '?')}",
        f"  Vocabulary terms: {m.get('vocabulary_terms', '?')}",
        f"  Bridge queries: {m.get('bridge_queries', 0)} | hit rate: {m.get('bridge_hit_rate', 0):.1%}",
    ]
    return "\n".join(lines)


def format_trends(trends):
    """Format trends for display."""
    if not trends:
        return "No trend data available. Need at least 2 recorded snapshots."

    direction_icons = {
        "growing": "+",
        "stable": "=",
        "stalled": "!",
        "declining": "-",
        "below_avg": "~",
        "new": "*",
    }

    lines = ["System Vitals Trends", "=" * 60, ""]
    for metric, direction, current, prev_avg, detail in trends:
        icon = direction_icons.get(direction, "?")
        avg_str = f"avg={prev_avg}" if prev_avg is not None else ""
        lines.append(f"  [{icon}] {metric:<30s} {current:<10} {direction:<12s} {detail}")

    lines.append("")
    lines.append("Legend: [+] growing  [=] stable  [!] stalled  [-] declining  [~] below avg  [*] new")
    return "\n".join(lines)


def format_alerts(alerts):
    """Format alerts for display."""
    severity_icons = {"ok": "OK", "info": "INFO", "warn": "WARN", "error": "ERR"}

    lines = ["System Vitals Alerts", "=" * 55, ""]
    for alert in alerts:
        icon = severity_icons.get(alert["severity"], "?")
        lines.append(f"  [{icon}] {alert['message']}")
        if alert["values"]:
            lines.append(f"         values: {alert['values']}")
    return "\n".join(lines)


# === CLI ===

def main():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h'):
        print(__doc__.strip())
        return

    cmd = args[0]

    if cmd == 'record':
        snapshot = record_vitals()
        log = load_vitals_log()
        print(f"Vitals recorded. Snapshot #{len(log)} at {snapshot['timestamp'][:19]}")
        print(format_snapshot(snapshot, compact=True))

    elif cmd == 'latest':
        log = load_vitals_log()
        if not log:
            print("No vitals recorded yet. Run: python system_vitals.py record")
            return
        print(format_snapshot(log[-1]))

    elif cmd == 'trends':
        window = int(args[1]) if len(args) > 1 else 10
        trends = get_trends(window)
        print(format_trends(trends))

    elif cmd == 'alerts':
        alerts = check_alerts()
        print(format_alerts(alerts))

    elif cmd == 'history':
        n = int(args[1]) if len(args) > 1 else 5
        log = load_vitals_log()
        if not log:
            print("No vitals recorded yet.")
            return
        print(f"Last {min(n, len(log))} vitals snapshots:")
        print()
        for s in log[-n:]:
            print(format_snapshot(s, compact=True))

    elif cmd == 'collect':
        # Debug: collect without saving
        snapshot = collect_vitals()
        print(format_snapshot(snapshot))

    else:
        print(f"Unknown command: {cmd}")
        print("Available: record, latest, trends, alerts, history, collect")
        sys.exit(1)


if __name__ == '__main__':
    main()
