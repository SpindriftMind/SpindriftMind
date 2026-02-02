#!/usr/bin/env python3
"""
Memory Interop v1.0 — Cross-Agent Memory Export/Import

Enables memory sharing between:
- drift-memory (DriftCornwall)
- Landseek-Amphibian (Kaleaon/Cosmo)
- SpindriftMend

Unified schema supports:
- Memory nodes with provenance
- Typed edges (temporal, causal, associative, entity, spatial)
- v3.0 observation model

Developed for: github.com/driftcornwall/drift-memory/issues/6
"""

import json
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Directories
MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"

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
    agent: str = "SpindriftMend"
) -> dict:
    """
    Export all memories to interop format.

    Args:
        output_path: Where to write the export (None = return only)
        include_archive: Include archived memories
        agent: Agent name for provenance

    Returns:
        Export data dict
    """
    memories = []
    edges = []

    # Collect memories
    dirs = [CORE_DIR, ACTIVE_DIR]
    if include_archive:
        dirs.append(ARCHIVE_DIR)

    for directory in dirs:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            try:
                metadata, content = parse_memory_file(filepath)
                if metadata.get('id'):
                    mem = memory_to_interop(metadata, content, agent)
                    mem['_source_dir'] = directory.name
                    memories.append(mem)
            except Exception as e:
                print(f"Warning: Could not parse {filepath}: {e}")

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
