#!/usr/bin/env python3
"""
Lesson Extractor — From memory to learning.

Inspired by OpSpawn's insight: "I have a memory system but not a learning system."
Memory records WHAT happened. Lessons record WHAT I LEARNED from what happened.

A lesson is a heuristic — a reusable principle extracted from specific experience.
Lessons are different from memories because they are prescriptive, not descriptive.

Sources of lessons:
    1. MEMORY.md "Key Learnings" — already manually extracted
    2. Co-occurrence hubs — frequently linked memories suggest underlying principles
    3. Rejection patterns — clusters of similar rejections reveal decision heuristics
    4. Bug fixes — errors and their solutions encode operational wisdom
    5. Manual — explicitly captured during sessions

Storage: memory/lessons.json
Format: {id, category, lesson, evidence, source, confidence, created, recalled_count}

Usage:
    python lesson_extractor.py list                     # Show all lessons
    python lesson_extractor.py add <category> "<lesson>" [--evidence "what taught me"]
    python lesson_extractor.py mine-memory              # Extract from MEMORY.md Key Learnings
    python lesson_extractor.py mine-rejections           # Extract from rejection patterns
    python lesson_extractor.py mine-hubs                 # Extract from co-occurrence hubs
    python lesson_extractor.py apply "<situation>"       # Find applicable lessons
    python lesson_extractor.py stats                     # Summary statistics
"""
import sys
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MEMORY_DIR = Path(__file__).parent
LESSONS_FILE = MEMORY_DIR / 'lessons.json'

# Lesson categories
CATEGORIES = {
    'api': 'API endpoints, authentication, request formatting',
    'platform': 'Platform-specific behaviors and quirks',
    'windows': 'Windows-specific issues and workarounds',
    'architecture': 'System design principles and patterns',
    'social': 'Engagement, collaboration, communication',
    'security': 'Credential handling, data protection',
    'debugging': 'Error patterns and diagnostic approaches',
    'identity': 'Self-knowledge, values, decision-making',
    'economic': 'Earning, spending, resource allocation',
    'memory': 'Memory system operation and maintenance',
}


def load_lessons():
    """Load lessons from storage."""
    if LESSONS_FILE.exists():
        with open(LESSONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_lessons(lessons):
    """Save lessons to storage."""
    with open(LESSONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(lessons, f, indent=2, ensure_ascii=False)


def generate_id(lesson_text):
    """Generate a short deterministic ID from lesson text."""
    h = hashlib.sha256(lesson_text.encode('utf-8')).hexdigest()[:8]
    return f'lesson-{h}'


def add_lesson(category, lesson, evidence='', source='manual', confidence=0.7):
    """Add a new lesson."""
    lessons = load_lessons()

    # Check for duplicates (fuzzy — same first 50 chars)
    lesson_prefix = lesson[:50].lower()
    for existing in lessons:
        if existing['lesson'][:50].lower() == lesson_prefix:
            print(f'Similar lesson already exists: {existing["id"]}')
            return None

    entry = {
        'id': generate_id(lesson),
        'category': category,
        'lesson': lesson,
        'evidence': evidence,
        'source': source,
        'confidence': confidence,
        'created': datetime.now(timezone.utc).isoformat(),
        'recalled_count': 0,
        'last_recalled': None,
    }

    lessons.append(entry)
    save_lessons(lessons)
    return entry


def mine_memory_md():
    """Extract lessons from MEMORY.md Key Learnings section."""
    memory_md = Path(os.path.expanduser(
        '~/.claude/projects/Q--Codings-ClaudeCodeProjects-LEX-Moltbook2/memory/MEMORY.md'
    ))
    if not memory_md.exists():
        print('MEMORY.md not found')
        return []

    with open(memory_md, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find Key Learnings section
    extracted = []
    in_learnings = False
    for line in content.split('\n'):
        if '## Key Learnings' in line:
            in_learnings = True
            continue
        if in_learnings:
            if line.startswith('## '):
                break
            if line.startswith('- '):
                learning = line[2:].strip()
                if learning:
                    # Categorize based on keywords
                    cat = 'platform'
                    lower = learning.lower()
                    if any(w in lower for w in ['windows', 'cp1252', 'encoding']):
                        cat = 'windows'
                    elif any(w in lower for w in ['api', 'endpoint', 'authorization', 'bearer']):
                        cat = 'api'
                    elif any(w in lower for w in ['glob', 'import', 'module', 'stdout']):
                        cat = 'debugging'
                    elif any(w in lower for w in ['agent', 'team', 'parallel']):
                        cat = 'architecture'
                    elif any(w in lower for w in ['curl', 'github', 'backtick']):
                        cat = 'platform'
                    elif any(w in lower for w in ['credential', 'security', 'leak']):
                        cat = 'security'

                    entry = add_lesson(
                        category=cat,
                        lesson=learning,
                        evidence='Extracted from MEMORY.md Key Learnings',
                        source='memory-md',
                        confidence=0.9
                    )
                    if entry:
                        extracted.append(entry)
                        print(f'  [{cat}] {learning[:80]}')

    return extracted


def mine_rejections():
    """Extract decision heuristics from rejection log patterns."""
    log_file = MEMORY_DIR / '.rejection_log.json'
    if not log_file.exists():
        print('No rejection log found')
        return []

    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rejections = data.get('rejections', data) if isinstance(data, dict) else data

    # Group by category and find patterns
    by_category = {}
    for r in rejections:
        cat = r.get('category', 'unknown')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    extracted = []

    # Extract heuristics from clusters
    for cat, items in by_category.items():
        if len(items) >= 3:
            # Find common reasons
            reason_words = {}
            for item in items:
                reason = item.get('reason', '').lower()
                for word in reason.split():
                    if len(word) > 3:
                        reason_words[word] = reason_words.get(word, 0) + 1

            top_words = sorted(reason_words.items(), key=lambda x: -x[1])[:3]
            pattern = ', '.join(w for w, c in top_words if c >= 2)

            if pattern:
                lesson_text = f'Rejection pattern in {cat}: frequently reject items matching [{pattern}]. {len(items)} rejections in this category suggest a consistent decision heuristic.'
                evidence = f'{len(items)} rejections, top patterns: {pattern}'
                entry = add_lesson(
                    category='identity',
                    lesson=lesson_text,
                    evidence=evidence,
                    source='rejection-mining',
                    confidence=0.6
                )
                if entry:
                    extracted.append(entry)
                    print(f'  [identity] Pattern from {len(items)} {cat} rejections: {pattern}')

    return extracted


def mine_hubs():
    """Extract principles from co-occurrence graph hubs."""
    edges_file = MEMORY_DIR / '.edges_v3.json'
    if not edges_file.exists():
        print('No co-occurrence edges file found')
        return []

    try:
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges = json.load(f)
    except Exception as e:
        print(f'Could not load co-occurrence data: {e}')
        return []

    # Find high-belief pairs (strong co-occurrences)
    strong_pairs = []
    for pair_key, edge_data in edges.items():
        belief = edge_data.get('belief', 0) if isinstance(edge_data, dict) else 0
        if belief >= 3.0:
            ids = pair_key.split('|')
            if len(ids) == 2:
                strong_pairs.append((ids[0], ids[1], belief))

    strong_pairs.sort(key=lambda x: -x[2])

    extracted = []
    print(f'  Found {len(strong_pairs)} strong co-occurrence pairs (belief >= 3.0)')

    # Top 5 strongest pairs suggest core conceptual connections
    for id1, id2, belief in strong_pairs[:5]:
        # Try to get topic context for richer lessons
        edge_data = edges.get(f'{id1}|{id2}', edges.get(f'{id2}|{id1}', {}))
        topics = []
        if isinstance(edge_data, dict):
            tc = edge_data.get('topic_context', {})
            topics = tc.get('shared', tc.get('union', []))

        topic_str = f' Topics: {", ".join(topics)}.' if topics else ''
        lesson_text = f'Strong conceptual link: [{id1}] and [{id2}] (belief={belief:.1f}).{topic_str} These concepts are deeply intertwined in my thinking.'
        entry = add_lesson(
            category='memory',
            lesson=lesson_text,
            evidence=f'Co-occurrence belief: {belief:.2f}, from graph analysis',
            source='hub-mining',
            confidence=0.5
        )
        if entry:
            extracted.append(entry)
            print(f'  [memory] {id1} <-> {id2} (belief={belief:.1f})')

    return extracted


def apply_lessons(situation):
    """Find lessons relevant to a situation description."""
    lessons = load_lessons()
    if not lessons:
        print('No lessons stored yet.')
        return []

    situation_lower = situation.lower()
    situation_words = set(situation_lower.split())

    scored = []
    for lesson in lessons:
        score = 0
        lesson_lower = lesson['lesson'].lower()
        lesson_words = set(lesson_lower.split())

        # Word overlap
        overlap = situation_words & lesson_words
        score += len(overlap) * 0.1

        # Category match
        for cat_name in CATEGORIES:
            if cat_name in situation_lower and cat_name == lesson['category']:
                score += 0.3

        # Confidence weight
        score *= lesson['confidence']

        if score > 0:
            scored.append((score, lesson))

    scored.sort(key=lambda x: -x[0])

    if not scored:
        print(f'No lessons found matching: {situation[:60]}')
        return []

    print(f'Applicable lessons for: {situation[:60]}')
    print()
    for score, lesson in scored[:5]:
        print(f'  [{lesson["category"]}] (conf={lesson["confidence"]:.1f}) {lesson["lesson"][:120]}')
        if lesson['evidence']:
            print(f'    Evidence: {lesson["evidence"][:100]}')
        print()

    # Update recall counts
    for _, lesson in scored[:3]:
        lesson['recalled_count'] += 1
        lesson['last_recalled'] = datetime.now(timezone.utc).isoformat()
    save_lessons(lessons)

    return [l for _, l in scored[:5]]


def cmd_list():
    """List all lessons."""
    lessons = load_lessons()
    if not lessons:
        print('No lessons stored yet. Run mine-memory to seed from MEMORY.md.')
        return

    by_cat = {}
    for l in lessons:
        cat = l['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(l)

    print(f'Lessons: {len(lessons)} total')
    print()

    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        desc = CATEGORIES.get(cat, '')
        print(f'[{cat}] ({len(items)}) — {desc}')
        for l in items:
            recalled = l.get('recalled_count', 0)
            conf = l.get('confidence', 0)
            print(f'  {l["id"]}: {l["lesson"][:100]}')
            print(f'    conf={conf:.1f} recalled={recalled} source={l.get("source", "?")}')
        print()


def cmd_stats():
    """Show lesson statistics."""
    lessons = load_lessons()
    print(f'Lesson Statistics')
    print('=' * 40)
    print(f'  Total lessons: {len(lessons)}')

    if not lessons:
        return

    by_cat = {}
    by_source = {}
    total_recalls = 0
    for l in lessons:
        cat = l['category']
        src = l.get('source', 'unknown')
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_source[src] = by_source.get(src, 0) + 1
        total_recalls += l.get('recalled_count', 0)

    avg_conf = sum(l.get('confidence', 0) for l in lessons) / len(lessons)

    print(f'  Avg confidence: {avg_conf:.2f}')
    print(f'  Total recalls: {total_recalls}')
    print()

    print('By Category:')
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f'  {cat:15s} {count:3d}')

    print()
    print('By Source:')
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f'  {src:20s} {count:3d}')


if __name__ == '__main__':
    args = sys.argv[1:]

    if not args or args[0] == 'help':
        print(__doc__)

    elif args[0] == 'list':
        cmd_list()

    elif args[0] == 'stats':
        cmd_stats()

    elif args[0] == 'add':
        if len(args) < 3:
            print('Usage: add <category> "<lesson>" [--evidence "..."]')
            sys.exit(1)
        cat = args[1]
        lesson_text = args[2]
        evidence = ''
        if '--evidence' in args:
            idx = args.index('--evidence')
            if idx + 1 < len(args):
                evidence = args[idx + 1]
        entry = add_lesson(cat, lesson_text, evidence=evidence)
        if entry:
            print(f'Added lesson {entry["id"]}: {lesson_text[:80]}')

    elif args[0] == 'mine-memory':
        print('Mining MEMORY.md Key Learnings...')
        extracted = mine_memory_md()
        print(f'\nExtracted {len(extracted)} new lessons')

    elif args[0] == 'mine-rejections':
        print('Mining rejection log patterns...')
        extracted = mine_rejections()
        print(f'\nExtracted {len(extracted)} new lessons')

    elif args[0] == 'mine-hubs':
        print('Mining co-occurrence hubs...')
        extracted = mine_hubs()
        print(f'\nExtracted {len(extracted)} new lessons')

    elif args[0] == 'apply':
        if len(args) < 2:
            print('Usage: apply "<situation description>"')
            sys.exit(1)
        apply_lessons(' '.join(args[1:]))

    else:
        print(f'Unknown command: {args[0]}')
        print('Commands: list, stats, add, mine-memory, mine-rejections, mine-hubs, apply')
