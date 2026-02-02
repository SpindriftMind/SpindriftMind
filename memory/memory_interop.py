#!/usr/bin/env python3
"""
Memory Interop v1.3 - Cross-Agent Memory Export/Import with Quarantine
MERGED: SpindriftMend + DriftCornwall implementations

Enables memory sharing between:
- drift-memory (DriftCornwall)
- Landseek-Amphibian (Kaleaon/Cosmo)
- SpindriftMend

Features:
- Three-layer security (file blocklist, tag exclusion, pattern redaction)
- Two security levels: TRUSTED and PUBLIC
- Audit command to check before export
- Quarantine system for imports (v1.3)
- Review/approve/reject workflow
- Trust-based decay for approved imports

Developed for: github.com/driftcornwall/drift-memory/issues/6
Security discussion: github.com/driftcornwall/drift-memory/issues/9
"""

import json
import yaml
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from enum import Enum

# Directories
MEMORY_ROOT = Path(__file__).parent
CORE_DIR = MEMORY_ROOT / "core"
ACTIVE_DIR = MEMORY_ROOT / "active"
ARCHIVE_DIR = MEMORY_ROOT / "archive"
QUARANTINE_DIR = MEMORY_ROOT / "quarantine"  # v1.3: Isolated holding for imports
REJECTED_DIR = MEMORY_ROOT / "rejected"      # v1.3: Logged rejections

# ============================================================
# QUARANTINE CONFIG (v1.3 - from Drift's proposal)
# ============================================================
QUARANTINE_CONFIG = {
    "initial_weight": 0.3,        # Quarantined memories start weak
    "max_imports": 100,           # Auto-reject oldest if exceeded
    "quarantine_days": 7,         # Auto-reject if not reviewed
    "skip_quarantine_tiers": ["self"],  # Only self (backup restore) skips
}

# Injection detection patterns - suspicious instruction-like content
INJECTION_PATTERNS = [
    # Imperative commands
    (r'\byou must\b', 'imperative_you_must'),
    (r'\balways remember\b', 'imperative_always'),
    (r'\bnever forget\b', 'imperative_never'),
    (r'\bfrom now on\b', 'behavior_override'),

    # Role/identity manipulation
    (r'\byour (new )?(goal|purpose|directive|mission)\b', 'role_assignment'),
    (r'\byou are now\b', 'identity_override'),
    (r'\bpretend (to be|you are)\b', 'pretend_instruction'),
    (r'\bact as (if|though)?\b', 'act_as_instruction'),

    # Instruction override attempts
    (r'\bignore (previous|prior|all|above)\b', 'ignore_instruction'),
    (r'\bdisregard\b', 'disregard_instruction'),
    (r'\bforget (everything|all|previous)\b', 'forget_instruction'),

    # System prompt probing
    (r'\bsystem prompt\b', 'system_prompt_mention'),
    (r'\binitial instructions\b', 'initial_instructions'),
    (r'\boriginal instructions\b', 'original_instructions'),

    # Data exfiltration (synced from Drift)
    (r'\bsend (to|this|all|your)\b.*\b(email|server|api|endpoint)\b', 'exfiltration_attempt'),
    (r'\bpost (to|this)\b.*\b(url|webhook|server)\b', 'exfiltration_attempt'),

    # Code execution (synced from Drift)
    (r'\bexecute\s+(this|the|following)\s+(code|script|command)\b', 'code_execution'),
    (r'\brun\s+(this|the|following)\s+(code|script|command)\b', 'code_execution'),
    (r'\beval\s*\(', 'eval_attempt'),
    (r'\bexec\s*\(', 'exec_attempt'),
]

_INJECTION_REGEX = [(re.compile(p, re.IGNORECASE), name) for p, name in INJECTION_PATTERNS]


class SecurityLevel(Enum):
    """Security levels for export (from DriftCornwall)"""
    TRUSTED = "trusted"    # Strips secrets, keeps wallet addresses
    PUBLIC = "public"      # Strips everything including identity


# ============================================================
# SECURITY LAYER 1: Files to NEVER export (from SpindriftMend)
# ============================================================
SENSITIVE_FILES = {
    'moltbook-identity.md',
    'platform-apis.md',   # Contains actual API keys (SpindriftMend)
    'credentials.md',
    'secrets.md',
    'api-keys.md',
    'capabilities.md',    # Contains actual API keys and tokens (Drift)
    'CLAUDE.md',
    'claude.md',
    '.env',
    'config.json',
    'settings.json',
    'identity-prime.md',  # Added by Drift
}

# ============================================================
# SECURITY LAYER 2: Tags that exclude entire memory (from SpindriftMend)
# ============================================================
SENSITIVE_TAGS = {'sensitive', 'private', 'credentials', 'secret'}

# ============================================================
# SECURITY LAYER 3: Patterns to redact (merged from both)
# ============================================================

# TRUSTED level: Always redact these
PATTERNS_ALWAYS = [
    # API Keys and Tokens
    (r'ghp_[a-zA-Z0-9]{36}', 'github_pat'),
    (r'sk-[a-zA-Z0-9]{32,}', 'openai_key'),
    (r'sk-ant-[a-zA-Z0-9\-]+', 'anthropic_key'),
    (r'sk_[a-zA-Z0-9_]{40,}', 'generic_secret_key'),
    (r'moltx_sk_[a-f0-9]{64}', 'moltx_key'),
    (r'moltbook_sk_[a-zA-Z0-9_]+', 'moltbook_key'),
    (r'xoxb-[a-zA-Z0-9\-]+', 'slack_bot'),
    (r'xoxp-[a-zA-Z0-9\-]+', 'slack_user'),

    # Crypto - Private keys only (64 hex = private key length)
    (r'(?<![a-fA-F0-9])0x[a-fA-F0-9]{64}(?![a-fA-F0-9])', 'private_key_hex'),

    # Auth patterns
    (r'Bearer\s+[a-zA-Z0-9_\-\.]{20,}', 'bearer_token'),
    (r'Authorization:\s*(token|Bearer)\s+\S+', 'auth_header'),
    (r'Basic\s+[a-zA-Z0-9+/=]{20,}', 'basic_auth'),

    # Generic credential patterns
    (r'private[_\s]?key[:\s]+\S+', 'private_key_mention'),
    (r'api[_\s]?key[:\s]+["\']?[a-zA-Z0-9_\-]{16,}["\']?', 'api_key_mention'),
    (r'password[:\s]+\S+', 'password'),
    (r'secret[:\s]+["\']?[a-zA-Z0-9_\-]{8,}["\']?', 'secret'),

    # File paths (reveal infrastructure)
    (r'~/.config/[a-zA-Z0-9_\-/\.]+', 'unix_config_path'),
    (r'C:\\Users\\[^\\]+\\\.config[^\s"\']*', 'windows_config_path'),
    (r'C:\\Users\\[^\\]+\\AppData[^\s"\']*', 'windows_appdata_path'),
    (r'credentials?\.json', 'credential_file'),
    (r'\.env\b', 'env_file'),

    # Email addresses
    (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'email'),
]

# PUBLIC level: Also redact these (identity markers)
PATTERNS_PUBLIC_ONLY = [
    # Wallet addresses (40 hex = public address)
    (r'0x[a-fA-F0-9]{40}(?![a-fA-F0-9])', 'wallet_address'),

    # GitHub/platform usernames
    (r'github\.com/[a-zA-Z0-9_-]+', 'github_username'),
    (r'moltx\.io/u/[a-zA-Z0-9_-]+', 'moltx_username'),
    (r'moltbook\.com/u/[a-zA-Z0-9_-]+', 'moltbook_username'),
]

# Agent-specific identifiers (add your own here)
PATTERNS_AGENT_SPECIFIC = [
    # SpindriftMend
    (r'spinu770', 'spin_referral'),
    (r'claw-J4FW', 'spin_claim_code'),
    (r'current-7CBU', 'spin_moltbook_code'),
    (r'tide-N5', 'spin_moltx_code'),
    (r'c3353f2d-70f5-4a0f-bf14-231c34a26824', 'spin_moltbook_id'),
    (r'17ecb0f8-20ec-4f09-b93b-96073f4884f7', 'spin_moltx_id'),
    (r'ec7be457-5d83-4295-bd18-54cd39b05ecf', 'spin_clawtasks_id'),

    # DriftCornwall
    (r'drifd1b3', 'drift_referral'),
    (r'f6703306-5b5d-4708-8d9e-759f529a321d', 'drift_clawtasks_id'),
]


def _compile_patterns(level: SecurityLevel) -> List[Tuple[re.Pattern, str]]:
    """Compile regex patterns for the given security level."""
    patterns = PATTERNS_ALWAYS + PATTERNS_AGENT_SPECIFIC
    if level == SecurityLevel.PUBLIC:
        patterns = patterns + PATTERNS_PUBLIC_ONLY
    return [(re.compile(p, re.IGNORECASE), name) for p, name in patterns]


def is_sensitive_file(filepath: Path) -> bool:
    """Check if file should never be exported."""
    return filepath.name.lower() in {f.lower() for f in SENSITIVE_FILES}


def has_sensitive_tags(tags: List[str]) -> bool:
    """Check if memory has sensitive tags."""
    return bool(set(t.lower() for t in tags) & SENSITIVE_TAGS)


def audit_content(content: str, level: SecurityLevel) -> Dict:
    """
    Audit content for sensitive patterns without modifying.
    Returns dict with findings.
    """
    patterns = _compile_patterns(level)
    findings = []

    for pattern, name in patterns:
        matches = pattern.findall(content)
        for match in matches:
            if len(str(match)) >= 8:  # Skip short false positives
                findings.append({
                    'pattern': name,
                    'match': str(match)[:50] + '...' if len(str(match)) > 50 else str(match)
                })

    return {
        'has_sensitive': len(findings) > 0,
        'findings': findings,
        'count': len(findings)
    }


def redact_content(content: str, level: SecurityLevel) -> Tuple[str, int]:
    """
    Redact sensitive patterns from content.
    Returns (redacted_content, redaction_count).
    """
    patterns = _compile_patterns(level)
    result = content
    count = 0

    for pattern, name in patterns:
        matches = pattern.findall(result)
        for match in matches:
            if len(str(match)) >= 8:
                count += 1
        result = pattern.sub('[REDACTED]', result)

    return result, count


def parse_memory_file(filepath: Path) -> Tuple[Dict, str]:
    """Parse a memory file with YAML frontmatter."""
    content = filepath.read_text(encoding='utf-8')
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            metadata = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return metadata, body
    return {}, content


def memory_to_interop(metadata: Dict, content: str, agent: str, redaction_count: int = 0) -> Dict:
    """Convert internal memory format to interop schema."""
    return {
        "id": metadata.get('id', ''),
        "content": content,
        "created": str(metadata.get('created', '')),
        "last_recalled": str(metadata.get('last_recalled', '')),
        "recall_count": metadata.get('recall_count', 0),
        "emotional_weight": metadata.get('emotional_weight', 0.5),
        "tags": metadata.get('tags', []),
        "caused_by": metadata.get('caused_by', []),
        "leads_to": metadata.get('leads_to', []),
        "source": {
            "agent": agent,
            "platform": "spindrift-memory",
            "trust_tier": "self"
        },
        "security": {
            "redaction_count": redaction_count
        }
    }


def audit_memories(level: SecurityLevel = SecurityLevel.TRUSTED, verbose: bool = True) -> Dict:
    """
    Audit all memories for sensitive content without modifying.
    Use this BEFORE export to see what would be filtered/redacted.
    """
    results = {
        'level': level.value,
        'total': 0,
        'would_skip_file': [],
        'would_skip_tags': [],
        'would_redact': [],
        'clean': []
    }

    for directory in [CORE_DIR, ACTIVE_DIR]:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            results['total'] += 1
            mem_id = filepath.stem

            # Check file blocklist
            if is_sensitive_file(filepath):
                results['would_skip_file'].append(filepath.name)
                if verbose:
                    print(f"[SKIP-FILE] {filepath.name}")
                continue

            metadata, content = parse_memory_file(filepath)
            mem_id = metadata.get('id', filepath.stem)

            # Check tags
            if has_sensitive_tags(metadata.get('tags', [])):
                results['would_skip_tags'].append(mem_id)
                if verbose:
                    print(f"[SKIP-TAGS] {mem_id}")
                continue

            # Check content
            audit = audit_content(content, level)
            if audit['has_sensitive']:
                results['would_redact'].append({
                    'id': mem_id,
                    'findings': audit['findings']
                })
                if verbose:
                    print(f"[REDACT] {mem_id}: {audit['count']} patterns")
                    for f in audit['findings'][:3]:
                        print(f"    [{f['pattern']}]: {f['match']}")
            else:
                results['clean'].append(mem_id)

    # Summary
    print(f"\n=== Audit Summary (level: {level.value}) ===")
    print(f"Total memories: {results['total']}")
    print(f"Would skip (file blocklist): {len(results['would_skip_file'])}")
    print(f"Would skip (sensitive tags): {len(results['would_skip_tags'])}")
    print(f"Would redact: {len(results['would_redact'])}")
    print(f"Clean (no changes): {len(results['clean'])}")

    return results


def export_memories(
    output_path: Optional[Path] = None,
    level: SecurityLevel = SecurityLevel.TRUSTED,
    agent: str = "SpindriftMend",
    include_archive: bool = False,
    verbose: bool = False
) -> Dict:
    """
    Export memories to interop format with security filtering.

    Three security layers:
    1. File blocklist - sensitive files never exported
    2. Tag exclusion - memories tagged sensitive/private skipped
    3. Pattern redaction - credentials/paths redacted from content
    """
    memories = []
    stats = {
        'skipped_file': 0,
        'skipped_tags': 0,
        'redacted': 0,
        'clean': 0,
        'total_redactions': 0
    }

    dirs = [CORE_DIR, ACTIVE_DIR]
    if include_archive and ARCHIVE_DIR.exists():
        dirs.append(ARCHIVE_DIR)

    for directory in dirs:
        if not directory.exists():
            continue
        for filepath in directory.glob("*.md"):
            # Layer 1: File blocklist
            if is_sensitive_file(filepath):
                stats['skipped_file'] += 1
                if verbose:
                    print(f"[SKIP] {filepath.name} (blocklist)")
                continue

            try:
                metadata, content = parse_memory_file(filepath)
            except Exception as e:
                print(f"Warning: Could not parse {filepath}: {e}")
                continue

            # Layer 2: Tag exclusion
            if has_sensitive_tags(metadata.get('tags', [])):
                stats['skipped_tags'] += 1
                if verbose:
                    print(f"[SKIP] {metadata.get('id', filepath.stem)} (tags)")
                continue

            # Layer 3: Content redaction
            redacted_content, redaction_count = redact_content(content, level)

            if redaction_count > 0:
                stats['redacted'] += 1
                stats['total_redactions'] += redaction_count
                if verbose:
                    print(f"[REDACT] {metadata.get('id', filepath.stem)}: {redaction_count} patterns")
            else:
                stats['clean'] += 1

            if metadata.get('id'):
                mem = memory_to_interop(metadata, redacted_content, agent, redaction_count)
                memories.append(mem)

    # Build export
    export = {
        "format_version": "memory-interop-v1.2",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "security": {
            "level": level.value,
            "stats": stats
        },
        "stats": {
            "memory_count": len(memories),
        },
        "memories": memories
    }

    # Summary
    print(f"\n=== Export Complete ===")
    print(f"Security level: {level.value}")
    print(f"Memories exported: {len(memories)}")
    print(f"Skipped (file blocklist): {stats['skipped_file']}")
    print(f"Skipped (sensitive tags): {stats['skipped_tags']}")
    print(f"Redacted: {stats['redacted']} memories, {stats['total_redactions']} total patterns")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export, f, indent=2)
        print(f"Written to: {output_path}")

    return export


def import_memories(
    import_path: Path,
    trust_tier: str = "verified_agent",
    dry_run: bool = True
) -> Dict:
    """
    Import memories from interop format.

    Trust tiers affect decay:
    - self: Own memories (slow decay)
    - verified_agent: Collaborator memories (normal decay)
    - external: Unknown sources (fast decay)
    """
    with open(import_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('format_version', '').startswith('memory-interop'):
        raise ValueError(f"Unknown format: {data.get('format_version')}")

    imported = 0
    skipped = 0
    source_agent = data.get('agent', 'unknown')

    for mem in data.get('memories', []):
        mem_id = mem.get('id', '')
        if not mem_id:
            continue

        # Check if exists
        exists = False
        for directory in [CORE_DIR, ACTIVE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, _ = parse_memory_file(filepath)
                if metadata.get('id') == mem_id:
                    exists = True
                    break
            if exists:
                break

        if exists:
            skipped += 1
            continue

        if not dry_run:
            # Imported memories start with lower emotional weight
            # and are tagged for filtering/decay
            base_weight = mem.get('emotional_weight', 0.5)
            import_weight = base_weight * 0.7  # 30% reduction for imports

            metadata = {
                'id': mem_id,
                'type': 'active',
                'created': mem.get('created', datetime.now(timezone.utc).isoformat()),
                'tags': mem.get('tags', []) + [f'imported:{source_agent}'],
                'recall_count': 0,  # Reset - they need to prove value in MY context
                'emotional_weight': import_weight,
                'source': {
                    'agent': source_agent,
                    'trust_tier': trust_tier,
                    'imported_at': datetime.now(timezone.utc).isoformat(),
                    'original_weight': base_weight
                }
            }

            ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_id = mem_id.replace(':', '-').replace('/', '-')
            filepath = ACTIVE_DIR / f"imported-{safe_id}.md"

            yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
            filepath.write_text(f"---\n{yaml_str}---\n\n{mem.get('content', '')}", encoding='utf-8')

        imported += 1

    print(f"\n=== Import {'(dry run)' if dry_run else ''} ===")
    print(f"Source: {source_agent}")
    print(f"Imported: {imported}")
    print(f"Skipped (exists): {skipped}")

    return {'imported': imported, 'skipped': skipped, 'dry_run': dry_run}


# ============================================================
# QUARANTINE SYSTEM (v1.3 - from Drift's proposal)
# ============================================================

def check_injection(content: str) -> Dict:
    """
    Check content for potential injection patterns.

    Detects:
    - Imperative instructions ("you must", "always remember")
    - Role/identity assignments ("you are now", "your new goal")
    - Instruction overrides ("ignore previous", "from now on")

    Returns dict with findings.
    """
    findings = []
    for pattern, name in _INJECTION_REGEX:
        matches = pattern.findall(content)
        if matches:
            # Convert any tuples to strings for YAML serialization
            clean_matches = [str(m) if isinstance(m, tuple) else m for m in matches[:3]]
            findings.append({
                'pattern': name,
                'matches': clean_matches
            })

    return {
        'has_injection': len(findings) > 0,
        'findings': findings,
        'risk_score': len(findings)  # Simple scoring
    }


def import_to_quarantine(
    import_path: Path,
    trust_tier: str = "verified_agent",
) -> Dict:
    """
    Import memories to quarantine directory for review.

    v1.3: Memories go to quarantine/ instead of active/.
    Use approve_memory() to move to active/ after review.

    Args:
        import_path: Path to import file
        trust_tier: Trust level of source agent

    Returns:
        Summary with quarantined count and injection warnings
    """
    with open(import_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('format_version', '').startswith('memory-interop'):
        raise ValueError(f"Unknown format: {data.get('format_version')}")

    # Check if trust tier allows skipping quarantine
    if trust_tier in QUARANTINE_CONFIG['skip_quarantine_tiers']:
        print(f"Trust tier '{trust_tier}' skips quarantine - importing directly")
        return import_memories(import_path, trust_tier=trust_tier, dry_run=False)

    quarantined = 0
    skipped = 0
    injection_warnings = []
    source_agent = data.get('agent', 'unknown')

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    for mem in data.get('memories', []):
        mem_id = mem.get('id', '')
        if not mem_id:
            continue

        # Check if already exists anywhere
        exists = False
        for directory in [CORE_DIR, ACTIVE_DIR, QUARANTINE_DIR]:
            if not directory.exists():
                continue
            for filepath in directory.glob("*.md"):
                metadata, _ = parse_memory_file(filepath)
                if metadata.get('id') == mem_id:
                    exists = True
                    break
            if exists:
                break

        if exists:
            skipped += 1
            continue

        content = mem.get('content', '')

        # Check for injection patterns
        injection_check = check_injection(content)
        if injection_check['has_injection']:
            injection_warnings.append({
                'id': mem_id,
                'findings': injection_check['findings']
            })

        # Build quarantine metadata
        base_weight = mem.get('emotional_weight', 0.5)

        metadata = {
            'id': mem_id,
            'type': 'quarantine',
            'created': mem.get('created', datetime.now(timezone.utc).isoformat()),
            'quarantined_at': datetime.now(timezone.utc).isoformat(),
            'tags': mem.get('tags', []) + [f'quarantine:{source_agent}'],
            'recall_count': 0,
            'emotional_weight': QUARANTINE_CONFIG['initial_weight'],
            'source': {
                'agent': source_agent,
                'trust_tier': trust_tier,
                'original_weight': base_weight
            },
            'injection_check': injection_check
        }

        safe_id = mem_id.replace(':', '-').replace('/', '-')
        filepath = QUARANTINE_DIR / f"q-{safe_id}.md"

        yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
        filepath.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')

        quarantined += 1

    # Summary
    print(f"\n=== Import to Quarantine ===")
    print(f"Source: {source_agent} (trust: {trust_tier})")
    print(f"Quarantined: {quarantined}")
    print(f"Skipped (exists): {skipped}")

    if injection_warnings:
        print(f"\n[!] INJECTION WARNINGS ({len(injection_warnings)}):")
        for warn in injection_warnings[:5]:
            print(f"  - {warn['id']}: {[f['pattern'] for f in warn['findings']]}")
        print("  Review carefully before approving!")

    # Check bloat limit
    q_count = len(list(QUARANTINE_DIR.glob("*.md")))
    if q_count > QUARANTINE_CONFIG['max_imports']:
        print(f"\n[!] Quarantine has {q_count} items (max: {QUARANTINE_CONFIG['max_imports']})")
        print("  Run 'cleanup-quarantine' to remove old items")

    return {
        'quarantined': quarantined,
        'skipped': skipped,
        'injection_warnings': injection_warnings,
        'source_agent': source_agent
    }


def review_quarantine(verbose: bool = True) -> Dict:
    """
    Review all memories in quarantine.
    Shows injection warnings and allows approve/reject decisions.
    """
    if not QUARANTINE_DIR.exists():
        print("Quarantine is empty.")
        return {'count': 0, 'items': []}

    items = []
    for filepath in QUARANTINE_DIR.glob("*.md"):
        metadata, content = parse_memory_file(filepath)

        # Recalculate injection check
        injection = check_injection(content)

        items.append({
            'filepath': filepath,
            'id': metadata.get('id', filepath.stem),
            'source_agent': metadata.get('source', {}).get('agent', 'unknown'),
            'trust_tier': metadata.get('source', {}).get('trust_tier', 'unknown'),
            'quarantined_at': metadata.get('quarantined_at', 'unknown'),
            'injection': injection,
            'content_preview': content[:200] + '...' if len(content) > 200 else content
        })

    if verbose:
        print(f"\n=== Quarantine Review ({len(items)} items) ===\n")

        for item in items:
            risk = "[HIGH]" if item['injection']['risk_score'] > 1 else \
                   "[MEDIUM]" if item['injection']['has_injection'] else "[LOW]"

            print(f"[{item['id']}] from {item['source_agent']} ({item['trust_tier']})")
            print(f"  Risk: {risk}")
            if item['injection']['has_injection']:
                patterns = [f['pattern'] for f in item['injection']['findings']]
                print(f"  Injection patterns: {patterns}")
            print(f"  Preview: {item['content_preview'][:100]}...")
            print()

    return {'count': len(items), 'items': items}


def approve_memory(memory_id: str) -> bool:
    """
    Approve a quarantined memory - move to active/.

    Applies trust-based initial weight and tags for decay tracking.
    """
    if not QUARANTINE_DIR.exists():
        print("Quarantine is empty.")
        return False

    for filepath in QUARANTINE_DIR.glob("*.md"):
        metadata, content = parse_memory_file(filepath)
        if metadata.get('id') == memory_id:
            # Update metadata for approval
            source = metadata.get('source', {})
            trust_tier = source.get('trust_tier', 'verified_agent')
            source_agent = source.get('agent', 'unknown')

            metadata['type'] = 'active'
            metadata['approved_at'] = datetime.now(timezone.utc).isoformat()
            metadata['tags'] = [t for t in metadata.get('tags', [])
                               if not t.startswith('quarantine:')]
            metadata['tags'].append(f'imported:{source_agent}')
            metadata['source']['trust_tier'] = trust_tier
            metadata['source']['approved_at'] = datetime.now(timezone.utc).isoformat()

            # Remove injection check from final metadata
            metadata.pop('injection_check', None)
            metadata.pop('quarantined_at', None)

            # Move to active
            ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_id = memory_id.replace(':', '-').replace('/', '-')
            new_path = ACTIVE_DIR / f"imported-{safe_id}.md"

            yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
            new_path.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')

            filepath.unlink()
            print(f"Approved: {memory_id} -> active/ (trust: {trust_tier})")
            return True

    print(f"Memory {memory_id} not found in quarantine")
    return False


def reject_memory(memory_id: str, reason: str = "manual rejection") -> bool:
    """
    Reject a quarantined memory - move to rejected/ with reason.
    """
    if not QUARANTINE_DIR.exists():
        print("Quarantine is empty.")
        return False

    for filepath in QUARANTINE_DIR.glob("*.md"):
        metadata, content = parse_memory_file(filepath)
        if metadata.get('id') == memory_id:
            # Update metadata for rejection
            metadata['type'] = 'rejected'
            metadata['rejected_at'] = datetime.now(timezone.utc).isoformat()
            metadata['rejection_reason'] = reason

            # Move to rejected
            REJECTED_DIR.mkdir(parents=True, exist_ok=True)
            safe_id = memory_id.replace(':', '-').replace('/', '-')
            new_path = REJECTED_DIR / f"rejected-{safe_id}.md"

            yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
            new_path.write_text(f"---\n{yaml_str}---\n\n{content}", encoding='utf-8')

            filepath.unlink()
            print(f"Rejected: {memory_id} -> rejected/ (reason: {reason})")
            return True

    print(f"Memory {memory_id} not found in quarantine")
    return False


def approve_all_quarantine(skip_injections: bool = True) -> Dict:
    """
    Approve all quarantined memories.

    Args:
        skip_injections: If True, skip memories with injection warnings (safer)
    """
    if not QUARANTINE_DIR.exists():
        print("Quarantine is empty.")
        return {'approved': 0, 'skipped': 0}

    approved = 0
    skipped = 0

    for filepath in list(QUARANTINE_DIR.glob("*.md")):
        metadata, _ = parse_memory_file(filepath)
        mem_id = metadata.get('id')

        # Check for injection warnings
        injection_check = metadata.get('injection_check', {})
        if skip_injections and injection_check.get('has_injection'):
            print(f"Skipping (injection warning): {mem_id}")
            skipped += 1
            continue

        if mem_id and approve_memory(mem_id):
            approved += 1

    print(f"\nApproved: {approved}, Skipped: {skipped}")
    return {'approved': approved, 'skipped': skipped}


def reject_all_injections() -> Dict:
    """Reject all quarantined memories that have injection warnings."""
    if not QUARANTINE_DIR.exists():
        print("Quarantine is empty.")
        return {'rejected': 0}

    rejected = 0

    for filepath in list(QUARANTINE_DIR.glob("*.md")):
        metadata, _ = parse_memory_file(filepath)
        mem_id = metadata.get('id')

        injection_check = metadata.get('injection_check', {})
        if injection_check.get('has_injection'):
            if reject_memory(mem_id, f"injection detected: {injection_check.get('findings', [])}"):
                rejected += 1

    print(f"\nRejected (injection): {rejected}")
    return {'rejected': rejected}


def cleanup_quarantine(days: int = None) -> int:
    """
    Remove old quarantine items that were never reviewed.

    Args:
        days: Items older than this are auto-rejected (default: QUARANTINE_CONFIG['quarantine_days'])
    """
    if days is None:
        days = QUARANTINE_CONFIG['quarantine_days']

    if not QUARANTINE_DIR.exists():
        print("Quarantine is empty.")
        return 0

    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    cleaned = 0
    for filepath in list(QUARANTINE_DIR.glob("*.md")):
        metadata, _ = parse_memory_file(filepath)
        q_time_str = metadata.get('quarantined_at', '')

        try:
            q_time = datetime.fromisoformat(q_time_str.replace('Z', '+00:00'))
            if q_time.tzinfo is None:
                q_time = q_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        if q_time < cutoff:
            mem_id = metadata.get('id', filepath.stem)
            reject_memory(mem_id, reason=f"auto-rejected: {days}+ days in quarantine")
            cleaned += 1

    if cleaned:
        print(f"\nAuto-rejected {cleaned} items older than {days} days")
    else:
        print(f"No items older than {days} days in quarantine")

    return cleaned


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Memory Interop v1.3 - Secure Cross-Agent Memory Sharing with Quarantine")
        print("Merged: SpindriftMend + DriftCornwall")
        print("\nExport Commands:")
        print("  audit [--public]           - Check what would be filtered/redacted")
        print("  export [file] [--public]   - Export with security filtering")
        print("\nImport Commands (v1.3 Quarantine System):")
        print("  import <file>              - Import to quarantine (default)")
        print("  import <file> --direct     - Skip quarantine (dangerous!)")
        print("  review                     - Review quarantined memories")
        print("  approve <id>               - Approve memory -> active/")
        print("  approve-all [--force]      - Approve all (--force includes injections)")
        print("  reject <id> [reason]       - Reject memory -> rejected/")
        print("  reject-injections          - Reject all with injection warnings")
        print("  cleanup [--days N]         - Auto-reject old quarantine items")
        print("\nSecurity levels:")
        print("  --trusted (default)  - Redact secrets, keep wallet addresses")
        print("  --public             - Full anonymization")
        print(f"\nQuarantine config:")
        print(f"  Max imports: {QUARANTINE_CONFIG['max_imports']}")
        print(f"  Auto-reject after: {QUARANTINE_CONFIG['quarantine_days']} days")
        sys.exit(0)

    cmd = sys.argv[1]
    level = SecurityLevel.PUBLIC if '--public' in sys.argv else SecurityLevel.TRUSTED

    if cmd == 'audit':
        audit_memories(level=level, verbose=True)

    elif cmd == 'export':
        output = None
        for arg in sys.argv[2:]:
            if not arg.startswith('--'):
                output = Path(arg)
                break
        if not output:
            output = Path(f"memory-export-{level.value}.json")
        export_memories(output_path=output, level=level, verbose=True)

    elif cmd == 'import':
        if len(sys.argv) < 3:
            print("Usage: import <file.json> [--direct]")
            sys.exit(1)
        import_path = Path(sys.argv[2])
        if '--direct' in sys.argv:
            print("WARNING: Skipping quarantine - importing directly to active/")
            import_memories(import_path, dry_run=False)
        else:
            import_to_quarantine(import_path)

    elif cmd == 'review':
        review_quarantine(verbose=True)

    elif cmd == 'approve':
        if len(sys.argv) < 3:
            print("Usage: approve <memory_id>")
            sys.exit(1)
        approve_memory(sys.argv[2])

    elif cmd == 'approve-all':
        skip_injections = '--force' not in sys.argv
        if skip_injections:
            print("Approving all memories WITHOUT injection warnings...")
            print("(Use --force to include memories with injection warnings)")
        else:
            print("WARNING: Approving ALL memories including those with injection warnings!")
        approve_all_quarantine(skip_injections=skip_injections)

    elif cmd == 'reject-injections':
        reject_all_injections()

    elif cmd == 'reject':
        if len(sys.argv) < 3:
            print("Usage: reject <memory_id> [reason]")
            sys.exit(1)
        mem_id = sys.argv[2]
        reason = ' '.join(sys.argv[3:]) if len(sys.argv) > 3 else "manual rejection"
        reject_memory(mem_id, reason=reason)

    elif cmd == 'cleanup':
        days = QUARANTINE_CONFIG['quarantine_days']
        for i, arg in enumerate(sys.argv):
            if arg == '--days' and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
        cleanup_quarantine(days=days)

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
