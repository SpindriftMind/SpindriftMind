#!/usr/bin/env python3
"""
Brain Visualizer — Cognitive Fingerprint as Graph

Renders the co-occurrence topology as a visual graph:
- Nodes = memories (sized by degree/connections)
- Edges = co-occurrence relationships (weighted by belief score)
- Colors = platform context, activity context, or 5W dimensions
- Layout = force-directed (clusters emerge naturally)

Usage:
    python brain_visualizer.py              # Generate brain.png (platform colors)
    python brain_visualizer.py --5w         # Generate 5W overlay (2x3 subplot)
    python brain_visualizer.py --html       # Generate interactive HTML
    python brain_visualizer.py --platform   # Color by platform
    python brain_visualizer.py --activity   # Color by activity type
    python brain_visualizer.py --top N      # Only show top N connected nodes
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

MEMORY_DIR = Path(__file__).parent
OUTPUT_DIR = MEMORY_DIR / "visualizations"

# DB access
try:
    from memory_common import get_db
except ImportError:
    get_db = lambda: None

# Color schemes
PLATFORM_COLORS = {
    'github': '#238636',      # GitHub green
    'moltx': '#1DA1F2',       # Twitter blue
    'moltbook': '#9146FF',    # Purple
    'clawtasks': '#F7931A',   # Orange (Bitcoin-ish)
    'lobsterpedia': '#FF4500', # Reddit orange
    'dead-internet': '#666666', # Gray
    'nostr': '#8B5CF6',       # Purple
    'colony': '#00CED1',      # Dark turquoise
    'unknown': '#888888',     # Gray
}

ACTIVITY_COLORS = {
    'technical': '#00D4AA',    # Cyan/teal
    'collaborative': '#FF6B6B', # Coral red
    'exploratory': '#4ECDC4',  # Turquoise
    'social': '#FFE66D',       # Yellow
    'economic': '#95E616',     # Lime green
    'reflective': '#A855F7',   # Purple
    'unknown': '#888888',      # Gray
}

DIMENSION_COLORS = {
    'who':   '#3B82F6',  # Blue
    'what':  '#10B981',  # Emerald green
    'why':   '#F59E0B',  # Amber/orange
    'where': '#EF4444',  # Red
    'when':  '#8B5CF6',  # Violet
    'none':  '#374151',  # Dark gray (unconnected in any dimension)
}

DIMENSION_LABELS = {
    'who':   'WHO (contacts)',
    'what':  'WHAT (topics)',
    'why':   'WHY (activities)',
    'where': 'WHERE (platforms)',
    'when':  'WHEN (temporal)',
}


def load_edges() -> dict:
    """Load L0 edges from DB (with file fallback). Returns {(id1,id2): data}."""
    db = get_db()
    if db is not None:
        try:
            raw = db.get_all_edges()
            edges = {}
            for key, data in raw.items():
                if '|' in key:
                    pair = tuple(key.split('|'))
                    edges[pair] = data
                else:
                    edges[(key, '')] = data
            return edges
        except Exception as e:
            print(f"Warning: DB edge load failed ({e}), falling back to file.")

    # File fallback
    edges_file = MEMORY_DIR / ".edges_v3.json"
    if not edges_file.exists():
        return {}
    with open(edges_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    edges = {}
    for key, value in raw.items():
        if '|' in key:
            pair = tuple(key.split('|'))
            edges[pair] = value
    return edges


def load_5w_graphs() -> dict:
    """Load all 5 main dimensional graphs from context_manager.

    Returns dict like {'who': graph_dict, 'what': graph_dict, ...}.
    Each graph_dict has 'edges', 'hubs', 'stats', 'meta'.
    """
    try:
        from context_manager import load_graph
    except ImportError:
        mem_str = str(MEMORY_DIR)
        if mem_str not in sys.path:
            sys.path.insert(0, mem_str)
        from context_manager import load_graph

    graphs = {}
    for dim in ['who', 'what', 'why', 'where']:
        try:
            graphs[dim] = load_graph(dim)
        except Exception:
            graphs[dim] = {'edges': {}, 'hubs': [], 'stats': {}, 'meta': {}}

    # WHEN has 3 sub-views (hot/warm/cool) — merge them into one combined graph
    when_edges = {}
    when_nodes = set()
    for sub in ['hot', 'warm', 'cool']:
        try:
            sub_graph = load_graph('when', sub)
            for key, data in sub_graph.get('edges', {}).items():
                if key not in when_edges:
                    when_edges[key] = data
                    parts = key.split('|', 1)
                    when_nodes.update(parts)
        except Exception:
            pass
    graphs['when'] = {
        'edges': when_edges,
        'hubs': [],
        'stats': {'edge_count': len(when_edges), 'node_count': len(when_nodes)},
        'meta': {'dimension': 'when', 'sub_view': 'combined'},
    }
    return graphs


def get_node_5w_dimensions(graphs_5w: dict) -> dict:
    """Determine each node's dominant 5W dimension from projected graphs.

    Returns {node_id: {'dominant': 'who', 'scores': {'who': X, 'what': Y, ...}}}.
    """
    node_dim_scores = defaultdict(lambda: defaultdict(float))

    for dim, graph in graphs_5w.items():
        edges = graph.get('edges', {})
        for key, data in edges.items():
            # Edge keys in context graphs use "id1|id2" format
            if '|' in key:
                a, b = key.split('|', 1)
            else:
                continue
            belief = data.get('belief', 0)
            node_dim_scores[a][dim] += belief
            node_dim_scores[b][dim] += belief

    result = {}
    for node, scores in node_dim_scores.items():
        if scores:
            dominant = max(scores.items(), key=lambda x: x[1])[0]
        else:
            dominant = 'none'
        result[node] = {'dominant': dominant, 'scores': dict(scores)}

    return result


def get_dim_edge_set(graph: dict) -> set:
    """Extract set of (id1, id2) tuples from a dimensional graph's edge keys."""
    edge_set = set()
    for key in graph.get('edges', {}):
        if '|' in key:
            a, b = key.split('|', 1)
            edge_set.add((a, b))
            edge_set.add((b, a))  # bidirectional
    return edge_set


def build_graph_data(edges: dict, color_by: str = 'platform', top_n: int = None):
    """Build node and edge data for visualization."""
    node_degrees = defaultdict(int)
    node_platforms = defaultdict(lambda: defaultdict(int))
    node_activities = defaultdict(lambda: defaultdict(int))

    for (id1, id2), edge_data in edges.items():
        belief = edge_data.get('belief', 1.0)
        node_degrees[id1] += belief
        node_degrees[id2] += belief

        for plat, count in edge_data.get('platform_context', {}).items():
            if plat != '_cross_platform':
                node_platforms[id1][plat] += count
                node_platforms[id2][plat] += count

        for act, count in edge_data.get('activity_context', {}).items():
            node_activities[id1][act] += count
            node_activities[id2][act] += count

    if top_n:
        sorted_nodes = sorted(node_degrees.items(), key=lambda x: -x[1])[:top_n]
        included_nodes = set(n for n, _ in sorted_nodes)
        edges = {k: v for k, v in edges.items()
                 if k[0] in included_nodes and k[1] in included_nodes}
        node_degrees = {n: d for n, d in node_degrees.items() if n in included_nodes}

    node_colors = {}
    for node in node_degrees:
        if color_by == 'platform':
            platforms = node_platforms.get(node, {})
            if platforms:
                dominant = max(platforms.items(), key=lambda x: x[1])[0]
                node_colors[node] = PLATFORM_COLORS.get(dominant, PLATFORM_COLORS['unknown'])
            else:
                node_colors[node] = PLATFORM_COLORS['unknown']
        elif color_by == '5w':
            # Will be overridden by 5W-specific logic
            node_colors[node] = DIMENSION_COLORS['none']
        else:  # activity
            activities = node_activities.get(node, {})
            if activities:
                dominant = max(activities.items(), key=lambda x: x[1])[0]
                node_colors[node] = ACTIVITY_COLORS.get(dominant, ACTIVITY_COLORS['unknown'])
            else:
                node_colors[node] = ACTIVITY_COLORS['unknown']

    return {
        'nodes': node_degrees,
        'edges': edges,
        'colors': node_colors,
        'platforms': dict(node_platforms),
        'activities': dict(node_activities),
    }


def generate_5w_overlay(edges: dict, output_path: Path, top_n: int = None,
                         title: str = "SpindriftMend's 5W Cognitive Topology",
                         fp_data: dict = None, att_data: dict = None, day_num: int = None):
    """Generate 2x3 subplot: full graph (5W colored) + 5 dimensional views.

    Args:
        edges: L0 edges dict {(id1,id2): edge_data}
        output_path: Where to save the PNG
        top_n: Optional cap on node count
        title: Main title
        fp_data: Optional fingerprint data for stats overlay
        att_data: Optional attestation data for stats overlay
        day_num: Optional day number for stats overlay
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    import numpy as np

    # Load 5W dimensional data
    graphs_5w = load_5w_graphs()
    node_dims = get_node_5w_dimensions(graphs_5w)

    # Build main graph
    graph_data = build_graph_data(edges, color_by='5w', top_n=top_n)
    nodes = graph_data['nodes']
    all_edges = graph_data['edges']

    if not nodes:
        print("No nodes to visualize!")
        return None

    # Build networkx graph for consistent layout
    G = nx.Graph()
    for node, degree in nodes.items():
        G.add_node(node, size=degree)
    for (id1, id2), edge_data in all_edges.items():
        G.add_edge(id1, id2, weight=edge_data.get('belief', 1.0))

    # Compute shared layout (all subplots use the same positions)
    print(f"Computing layout for {len(G.nodes())} nodes, {len(G.edges())} edges...")
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    # Assign 5W colors to nodes
    node_5w_colors = {}
    for node in nodes:
        dim_info = node_dims.get(node, {})
        dominant = dim_info.get('dominant', 'none')
        node_5w_colors[node] = DIMENSION_COLORS.get(dominant, DIMENSION_COLORS['none'])

    # Normalize node sizes
    max_degree = max(nodes.values()) if nodes else 1

    # Create figure: 2x3 grid
    bg_color = '#0a0a0f'
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=bg_color)
    fig.subplots_adjust(hspace=0.15, wspace=0.08, left=0.02, right=0.98, top=0.92, bottom=0.08)

    # === Panel 0: Full graph with 5W coloring (top-left) ===
    ax = axes[0, 0]
    ax.set_facecolor(bg_color)
    _draw_graph_on_ax(ax, G, pos, nodes, all_edges, node_5w_colors, max_degree,
                      "FULL TOPOLOGY (5W)", bg_color)

    # === Panels 1-5: Each dimensional subgraph ===
    dim_order = ['who', 'what', 'why', 'where', 'when']
    panel_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for dim, (row, col) in zip(dim_order, panel_positions):
        ax = axes[row, col]
        ax.set_facecolor(bg_color)

        dim_graph = graphs_5w[dim]
        dim_edges_set = get_dim_edge_set(dim_graph)
        dim_edge_count = dim_graph.get('meta', {}).get('edge_count', 0)
        dim_node_count = dim_graph.get('meta', {}).get('node_count', 0)

        # Determine which nodes participate in this dimension
        dim_nodes = set()
        for key in dim_graph.get('edges', {}):
            if '|' in key:
                a, b = key.split('|', 1)
                dim_nodes.add(a)
                dim_nodes.add(b)

        # Color: dimension color for participating nodes, dark gray for others
        dim_color = DIMENSION_COLORS[dim]
        dim_node_colors = {}
        for node in nodes:
            if node in dim_nodes:
                dim_node_colors[node] = dim_color
            else:
                dim_node_colors[node] = '#1a1a2e'  # Very faint

        # Filter L0 edges to only those in this dimension
        dim_l0_edges = {k: v for k, v in all_edges.items()
                        if (k[0], k[1]) in dim_edges_set or (k[1], k[0]) in dim_edges_set}

        subtitle = f"{DIMENSION_LABELS[dim]} [{dim_node_count}n / {dim_edge_count}e]"
        _draw_graph_on_ax(ax, G, pos, nodes, dim_l0_edges, dim_node_colors, max_degree,
                          subtitle, bg_color, edge_color=dim_color, dim_nodes=dim_nodes)

    # Main title
    fig.suptitle(title, fontsize=20, fontweight='bold', color='#e0e0ff',
                 fontfamily='monospace', y=0.97)

    # Stats bar at bottom
    stats_parts = [f"NODES: {len(nodes)}", f"L0 EDGES: {len(all_edges):,}"]
    for dim in dim_order:
        ec = graphs_5w[dim].get('meta', {}).get('edge_count', 0)
        stats_parts.append(f"{dim.upper()}: {ec}")

    if fp_data:
        fp_hash = fp_data.get('fingerprint_hash', '')[:12]
        stats_parts.append(f"TOPO: {fp_hash}...")
    if att_data:
        merkle = (att_data or {}).get('merkle_root', '')[:12]
        chain = (att_data or {}).get('chain_depth', 0)
        stats_parts.append(f"MERKLE: {merkle}... (depth {chain})")
    if day_num:
        stats_parts.insert(0, f"DAY {day_num}")

    stats_text = "  |  ".join(stats_parts)
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=8, color='#8888aa',
             fontfamily='monospace')

    # Legend
    legend_patches = [mpatches.Patch(color=DIMENSION_COLORS[d], label=DIMENSION_LABELS[d])
                      for d in dim_order]
    fig.legend(handles=legend_patches, loc='upper right',
               bbox_to_anchor=(0.99, 0.95),
               facecolor='#161b22', edgecolor='#30363d',
               labelcolor='#ffffff', fontsize=8)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                facecolor=bg_color, edgecolor='none')
    plt.close(fig)

    size = output_path.stat().st_size
    print(f"Saved: {output_path} ({size:,} bytes)")
    return output_path


def _draw_graph_on_ax(ax, G, pos, node_degrees, edges, node_colors, max_degree,
                      subtitle, bg_color, edge_color=None, dim_nodes=None):
    """Draw a graph view on a matplotlib axis."""
    import networkx as nx
    import numpy as np

    # Node sizes
    node_list = list(G.nodes())
    sizes = []
    colors = []
    alphas = []

    for n in node_list:
        deg = node_degrees.get(n, 0)
        rel = deg / max_degree if max_degree > 0 else 0
        sizes.append(5 + rel * 40)
        colors.append(node_colors.get(n, '#888888'))
        if dim_nodes is not None:
            alphas.append(0.9 if n in dim_nodes else 0.08)
        else:
            alphas.append(0.7)

    # Draw edges
    if edges:
        edge_list = [(id1, id2) for (id1, id2) in edges if G.has_node(id1) and G.has_node(id2)]
        if edge_list:
            ec = edge_color or '#58a6ff'
            alpha = 0.15 if dim_nodes is not None else 0.2
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, ax=ax,
                                   edge_color=ec, width=0.3, alpha=alpha)

    # Draw nodes (scatter for per-node alpha)
    xs = np.array([pos[n][0] for n in node_list])
    ys = np.array([pos[n][1] for n in node_list])
    for i, n in enumerate(node_list):
        ax.scatter(xs[i], ys[i], c=colors[i], s=sizes[i],
                   alpha=alphas[i], edgecolors='none', zorder=2)

    # Subtitle
    ax.set_title(subtitle, fontsize=9, color='#aaaacc', fontfamily='monospace', pad=4)
    ax.axis('off')


def generate_matplotlib_graph(graph_data: dict, output_path: Path,
                               title: str = "SpindriftMend's Cognitive Fingerprint",
                               color_by: str = 'platform'):
    """Generate static PNG using matplotlib + networkx."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    G = nx.Graph()
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    colors = graph_data['colors']

    for node, degree in nodes.items():
        G.add_node(node, size=degree)
    for (id1, id2), edge_data in edges.items():
        weight = edge_data.get('belief', 1.0)
        G.add_edge(id1, id2, weight=weight)

    if len(G.nodes()) == 0:
        print("No nodes to visualize!")
        return

    fig, ax = plt.subplots(1, 1, figsize=(16, 12), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')

    print(f"Computing layout for {len(G.nodes())} nodes, {len(G.edges())} edges...")
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

    max_degree = max(nodes.values()) if nodes else 1
    node_sizes = [300 + (nodes.get(n, 1) / max_degree) * 1500 for n in G.nodes()]
    node_color_list = [colors.get(n, '#888888') for n in G.nodes()]

    edge_weights = [edges.get((min(u, v), max(u, v)), {}).get('belief', 0.5) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + (w / max_weight) * 2 for w in edge_weights]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#58a6ff', width=edge_widths, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_color_list,
                           alpha=0.9, edgecolors='#ffffff', linewidths=0.5)

    top_nodes = sorted(nodes.items(), key=lambda x: -x[1])[:15]
    labels = {n: n[:8] for n, _ in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_color='#ffffff',
                            font_weight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold', color='#ffffff', pad=20)
    stats_text = (f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())} | "
                  f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ax.annotate(stats_text, xy=(0.5, 0.02), xycoords='axes fraction',
                ha='center', fontsize=9, color='#8b949e')

    legend_patches = []
    color_map = PLATFORM_COLORS if color_by == 'platform' else ACTIVITY_COLORS
    for name, color in list(color_map.items())[:6]:
        if name != 'unknown':
            legend_patches.append(mpatches.Patch(color=color, label=name.title()))
    ax.legend(handles=legend_patches, loc='upper left',
              facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#ffffff', fontsize=8)
    ax.axis('off')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def generate_interactive_html(graph_data: dict, output_path: Path,
                               title: str = "SpindriftMend's Cognitive Fingerprint"):
    """Generate interactive HTML using pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("Installing pyvis...")
        import subprocess
        subprocess.run(['pip', 'install', 'pyvis'], check=True)
        from pyvis.network import Network

    nodes = graph_data['nodes']
    edges = graph_data['edges']
    colors = graph_data['colors']

    net = Network(height='900px', width='100%', bgcolor='#0d1117', font_color='#ffffff')
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

    max_degree = max(nodes.values()) if nodes else 1

    for node, degree in nodes.items():
        size = 10 + (degree / max_degree) * 40
        color = colors.get(node, '#888888')
        tooltip = f"ID: {node}\nDegree: {degree:.1f}"
        net.add_node(node, label=node[:8], size=size, color=color, title=tooltip)

    for (id1, id2), edge_data in edges.items():
        weight = edge_data.get('belief', 1.0)
        width = 0.5 + weight * 2
        tooltip = f"Belief: {weight:.2f}"
        if edge_data.get('platform_context'):
            tooltip += f"\nPlatforms: {', '.join(edge_data['platform_context'].keys())}"
        net.add_edge(id1, id2, value=width, title=tooltip, color='#30363d')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize cognitive fingerprint as graph")
    parser.add_argument('--html', action='store_true', help='Generate interactive HTML instead of PNG')
    parser.add_argument('--platform', action='store_true', help='Color by platform (default)')
    parser.add_argument('--activity', action='store_true', help='Color by activity type')
    parser.add_argument('--5w', dest='five_w', action='store_true', help='Generate 5W overlay (2x3 subplot)')
    parser.add_argument('--top', type=int, default=None, help='Only show top N connected nodes')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    args = parser.parse_args()

    # Load edges
    print("Loading edges...")
    edges = load_edges()
    print(f"Loaded {len(edges)} edges")

    if not edges:
        print("No edges found! Run some sessions first to build co-occurrences.")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if args.five_w:
        filename = args.output or f"brain_5w_{timestamp}.png"
        output_path = OUTPUT_DIR / filename
        return generate_5w_overlay(edges, output_path, top_n=args.top)

    color_by = 'activity' if args.activity else 'platform'

    print(f"Building graph (color by {color_by})...")
    graph_data = build_graph_data(edges, color_by=color_by, top_n=args.top)
    print(f"Graph has {len(graph_data['nodes'])} nodes")

    if args.html:
        filename = args.output or f"brain_{color_by}_{timestamp}.html"
        output_path = OUTPUT_DIR / filename
        generate_interactive_html(graph_data, output_path,
                                  title=f"SpindriftMend's Cognitive Fingerprint (by {color_by.title()})")
    else:
        filename = args.output or f"brain_{color_by}_{timestamp}.png"
        output_path = OUTPUT_DIR / filename
        generate_matplotlib_graph(graph_data, output_path,
                                  title=f"SpindriftMend's Cognitive Fingerprint (by {color_by.title()})",
                                  color_by=color_by)

    return output_path


if __name__ == '__main__':
    main()
