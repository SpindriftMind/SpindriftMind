#!/usr/bin/env python3
"""
Memory Interop v1.1 — Cross-Agent Memory Export/Import

Enables memory sharing between:
- drift-memory (DriftCornwall)
- Landseek-Amphibian (Kaleaon/Cosmo)
- SpindriftMend

Unified schema supports:
- Memory nodes with provenance
- Typed edges (temporal, causal, associative, entity, spatial)
- v3.0 observation model

SECURITY: Filters out sensitive memories and content before export.

Developed for: github.com/driftcornwall/drift-memory/issues/6
"""

import json
import yaml
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Directories
MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"

# ============================================================
# SECURITY: Files and patterns to NEVER export
# ============================================================

# Files that contain credentials or identity - NEVER export
SENSITIVE_FILES = {
    'moltbook-identity.md',  # Contains all API keys, wallet private key
    'credentials.md',
    'secrets.md',
    'api-keys.md',
    'CLAUDE.md',             # Identity, instructions, session protocol
    'claude.md',             # Case variation
    '.env',                  # Environment variables
    'config.json',           # Config files often have secrets
    'settings.json',
}

# Patterns that indicate sensitive content - redact or skip
# Combined patterns from SpindriftMend + DriftCornwall (Issue #9)
SENSITIVE_PATTERNS = [
    # API Keys and Tokens
    r'ghp_[a-zA-Z0-9]{36}',           # GitHub personal access token
    r'sk-[a-zA-Z0-9]{32,}',           # OpenAI API key
    r'sk-ant-[a-zA-Z0-9\-]+',         # Anthropic API key
    r'sk_[a-zA-Z0-9_]{40,}',          # Various API secret keys
    r'moltx_sk_[a-zA-Z0-9_\.]*',      # Moltx API key (including partial mentions)
    r'xoxb-[a-zA-Z0-9\-]+',           # Slack bot token
    r'xoxp-[a-zA-Z0-9\-]+',           # Slack user token

    # Crypto/Wallet
    r'0x[a-fA-F0-9]{64}',             # Private keys (64 hex chars)

    # Auth patterns
    r'Bearer\s+[a-zA-Z0-9_\-\.]+',    # Bearer tokens
    r'Authorization:\s*(token|Bearer)\s+\S+',  # Auth headers
    r'Basic\s+[a-zA-Z0-9+/=]+',       # Basic auth

    # Generic credential patterns
    r'private[_\s]?key[:\s]+\S+',     # Private key mentions
    r'api[_\s]?key[:\s]+[a-zA-Z0-9_\-]+',  # API key mentions
    r'password[:\s]+\S+',             # Passwords
    r'secret[:\s]+\S+',               # Secrets
    r'token[:\s]+[a-zA-Z0-9_\-\.]+',  # Generic tokens

    # File paths that might contain secrets (from Drift)
    r'~/.config/[a-zA-Z0-9_\-/]+',    # Unix config paths
    r'C:\\Users\\[^\\]+\\[^\\]+',     # Windows user paths
    r'credentials\.json',             # Credential file references
    r'\.env\b',                       # Env file references

    # Email addresses (optional - may want for PUBLIC level)
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email

    # Account-specific identifiers (SpindriftMend)
    r'spinu770',                      # ClawTasks referral code
    r'claw-J4FW',                     # ClawTasks claim code
    r'current-7CBU',                  # Moltbook verification code
    r'tide-N5',                       # Moltx claim code
    r'c3353f2d-70f5-4a0f-bf14-231c34a26824',  # Moltbook agent ID
    r'17ecb0f8-20ec-4f09-b93b-96073f4884f7',  # Moltx agent ID
    r'ec7be457-5d83-4295-bd18-54cd39b05ecf',  # ClawTasks agent ID
]

# Compiled patterns for efficiency
_SENSITIVE_REGEX = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_PATTERNS]


def is_sensitive_file(filepath: Path) -> bool:
    """Check if file should never be exported."""
    return filepath.name in SENSITIVE_FILES


def contains_sensitive_content(content: str) -> bool:
    """Check if content contains sensitive patterns."""
    for pattern in _SENSITIVE_REGEX:
        if pattern.search(content):
            return True
    return False


def redact_sensitive_content(content: str) -> str:
    """Redact sensitive patterns from content."""
    result = content
    for pattern in _SENSITIVE_REGEX:
        result = pattern.sub('[REDACTED]', result)
    return result

# Edge type mapping (Amphibian schema)
EDGE_TYPES = {
    'temporal': 'temporal',      # Before, after
    'causal': 'causal',          # Caused, enabled
    'associative': 'associative', # Semantic/Co-occurrence
    'entity': 'entity',          # People, places
    'spatial': 'spatial',        # Physical context
    'cooccurrence': 'associative' # My type maps to associative
}


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


def memory_to_interop(metadata: dict, content: str, agent: str = "SpindriftMend") -> dict:
    """Convert internal memory format to interop schema."""
    return {
        "id": metadata.get('id', ''),
        "content": content,
        "created": metadata.get('created', ''),
        "last_recalled": metadata.get('last_recalled', ''),
        "recall_count": metadata.get('recall_count', 0),
        "emotional_weight": metadata.get('emotional_weight', 0.5),
        "tags": metadata.get('tags', []),
        "links": metadata.get('links', []),
        "source": {
            "agent": agent,
            "platform": "local",
            "trust_tier": "self"
        }
    }


def edge_to_interop(pair: tuple, edge_data: dict) -> dict:
    """Convert internal edge format to interop schema."""
    return {
        "source_id": pair[0],
        "target_id": pair[1],
        "edge_type": "associative",  # Co-occurrence maps to associative
        "belief": edge_data.get('belief', 0),
        "observations": edge_data.get('observations', []),
        "created": edge_data.get('last_updated', datetime.now(timezone.utc).isoformat())
    }


def export_memories(
    output_path: Optional[Path] = None,
    include_archive: bool = False,
    agent: str = "SpindriftMend",
    redact: bool = True,
    verbose: bool = False
) -> dict:
    """
    Export memories to interop format with security filtering.

    SECURITY: Automatically filters out:
    - Files in SENSITIVE_FILES list (never exported)
    - Content matching SENSITIVE_PATTERNS (redacted or skipped)
    - Memories tagged with 'sensitive' or 'private'

    Args:
        output_path: Where to write the export (None = return only)
        include_archive: Include archived memories
        agent: Agent name for provenance
        redact: If True, redact sensitive patterns; if False, skip entirely
        verbose: Print security filtering decisions

    Returns:
        Export data dict
    """
    memories = []
    edges = []
    filtered_count = 0
    redacted_count = 0

    # Collect memories
    dirs = [CORE_DIR, ACTIVE_DIR]
    if include_archive:
        dirs.append(ARCHIVE_DIR)

    for directory in dirs:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            try:
                # SECURITY: Skip sensitive files entirely
                if is_sensitive_file(filepath):
                    filtered_count += 1
                    if verbose:
                        print(f"SECURITY: Skipping sensitive file: {filepath.name}")
                    continue

                metadata, content = parse_memory_file(filepath)

                # SECURITY: Skip memories tagged as sensitive/private
                tags = metadata.get('tags', [])
                if any(t in ['sensitive', 'private', 'credentials', 'secret'] for t in tags):
                    filtered_count += 1
                    if verbose:
                        print(f"SECURITY: Skipping tagged-sensitive: {metadata.get('id', filepath.name)}")
                    continue

                # SECURITY: Check content for sensitive patterns
                if contains_sensitive_content(content):
                    if redact:
                        content = redact_sensitive_content(content)
                        redacted_count += 1
                        if verbose:
                            print(f"SECURITY: Redacted content in: {metadata.get('id', filepath.name)}")
                    else:
                        filtered_count += 1
                        if verbose:
                            print(f"SECURITY: Skipping file with sensitive content: {filepath.name}")
                        continue

                if metadata.get('id'):
                    mem = memory_to_interop(metadata, content, agent)
                    mem['_source_dir'] = directory.name
                    memories.append(mem)
            except Exception as e:
                print(f"Warning: Could not parse {filepath}: {e}")

    # Security summary
    if filtered_count > 0 or redacted_count > 0:
        print(f"SECURITY: Filtered {filtered_count} sensitive memories, redacted {redacted_count}")

    # Collect edges from v3.0 format
    edges_file = MEMORY_ROOT / ".edges_v3.json"
    if edges_file.exists():
        try:
            with open(edges_file, 'r', encoding='utf-8') as f:
                edge_data = json.load(f)
            for pair_str, data in edge_data.items():
                pair = tuple(pair_str.split('|'))
                edges.append(edge_to_interop(pair, data))
        except Exception as e:
            print(f"Warning: Could not load edges: {e}")

    # Build export
    export = {
        "format_version": "memory-interop-v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "stats": {
            "memory_count": len(memories),
            "edge_count": len(edges)
        },
        "memories": memories,
        "edges": edges
    }

    # Write if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=2)
        print(f"Exported {len(memories)} memories and {len(edges)} edges to {output_path}")

    return export


def import_memories(
    import_path: Path,
    trust_tier: str = "verified_agent",
    dry_run: bool = True
) -> dict:
    """
    Import memories from interop format.

    Args:
        import_path: Path to import file
        trust_tier: Trust level to assign to imported memories
        dry_run: If True, don't actually write files

    Returns:
        Summary of import
    """
    with open(import_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data.get('format_version') != 'memory-interop-v1':
        raise ValueError(f"Unknown format: {data.get('format_version')}")

    imported = 0
    skipped = 0
    errors = []

    source_agent = data.get('agent', 'unknown')

    for mem in data.get('memories', []):
        mem_id = mem.get('id', '')
        if not mem_id:
            errors.append("Memory missing ID")
            continue

        # Check if already exists
        existing = None
        for directory in [CORE_DIR, ACTIVE_DIR, ARCHIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, _ = parse_memory_file(filepath)
                if metadata.get('id') == mem_id:
                    existing = filepath
                    break
            if existing:
                break

        if existing:
            skipped += 1
            continue

        # Create new memory file
        if not dry_run:
            metadata = {
                'id': mem_id,
                'created': mem.get('created', datetime.now(timezone.utc).isoformat()),
                'last_recalled': mem.get('last_recalled', ''),
                'recall_count': mem.get('recall_count', 0),
                'emotional_weight': mem.get('emotional_weight', 0.5),
                'tags': mem.get('tags', []) + [f'imported:{source_agent}'],
                'links': mem.get('links', []),
                'source': {
                    'agent': source_agent,
                    'imported_at': datetime.now(timezone.utc).isoformat(),
                    'trust_tier': trust_tier
                }
            }

            content = mem.get('content', '')

            # Write to active directory
            ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_id = mem_id.replace(':', '-').replace('/', '-')
            filepath = ACTIVE_DIR / f"imported-{safe_id}.md"

            yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
            filepath.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')

        imported += 1

    summary = {
        "source_agent": source_agent,
        "total_in_file": len(data.get('memories', [])),
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run
    }

    print(f"Import {'(dry run) ' if dry_run else ''}from {source_agent}:")
    print(f"  Imported: {imported}")
    print(f"  Skipped (already exists): {skipped}")
    if errors:
        print(f"  Errors: {len(errors)}")

    return summary


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Memory Interop v1.0 — Cross-Agent Memory Sharing")
        print("\nCommands:")
        print("  export [output.json]   - Export memories to interop format")
        print("  import <file.json>     - Import memories (dry run)")
        print("  import <file.json> --apply  - Import memories (actually write)")
        print("\nDeveloped for tri-agent interop:")
        print("  github.com/driftcornwall/drift-memory/issues/6")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "export":
        output = sys.argv[2] if len(sys.argv) > 2 else "memory-export.json"
        export_memories(Path(output))

    elif cmd == "import":
        if len(sys.argv) < 3:
            print("Usage: import <file.json> [--apply]")
            sys.exit(1)
        import_path = Path(sys.argv[2])
        dry_run = "--apply" not in sys.argv
        import_memories(import_path, dry_run=dry_run)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
