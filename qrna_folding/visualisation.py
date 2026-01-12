from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns


def plot_adjacency_matrix(adj_matrix: np.ndarray, rna_seq: str, ax=None):
    """
    Plots an adjacency matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.bitwise_not(np.triu(np.ones_like(adj_matrix, dtype=bool)))
    mat = adj_matrix
    np.fill_diagonal(mat, -1)
    sns.heatmap(
        mat,
        mask=mask,
        ax=ax,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.xaxis.set_ticks(ticks=range(len(rna_seq)), labels=list(rna_seq))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks(ticks=range(len(rna_seq)), labels=list(rna_seq))

    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-1, 0, 2, 3])
    colorbar.set_ticklabels(["NA", "NA", "AU/GU", "GC"])
    colorbar.ax.tick_params(labelsize=16)
    return ax


def generate_network_graph_image(rna: str, stem: Tuple[int, int, int], ax=None):
    """
    Generates a network graph for a given stem.
    """
    colors = {"G": "#FF0060", "A": "#F6FA70", "C": "#00DFA2", "U": "#0079FF"}
    n = len(rna)
    # Create base graph
    graph = nx.Graph()
    for i, base in enumerate(rna):
        graph.add_node(i, color=colors[base], label=base)
        if i + 1 < n:
            graph.add_edge(i, i + 1)

    # Add potential stem
    stem_span1 = list(range(stem[0] - 1, stem[0] + stem[2] - 1))
    stem_span2 = list(range(stem[1] - stem[2], stem[1]))
    stem_span2.reverse()

    for b1, b2 in zip(stem_span1, stem_span2):
        graph.add_edge(b1, b2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))

    node_colors = [graph.nodes[node]["color"] for node in graph.nodes]
    node_labels = {node: graph.nodes[node]["label"] for node in graph.nodes}
    nx.draw(
        graph,
        ax=ax,
        pos=nx.layout.kamada_kawai_layout(graph, scale=10),
        node_size=300,
        width=1,
        node_color=node_colors,
        style="-",
        labels=node_labels,
        with_labels=True,
        font_size=12,
    )

    return ax


def plot_convergence(costs: List[float], ax=None):
    """
    Plot QAOA optimization convergence (cost vs iteration).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(costs, linewidth=2, color="#0079FF")
    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_title("QAOA Convergence", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark minimum
    min_idx = np.argmin(costs)
    min_cost = costs[min_idx]
    ax.scatter([min_idx], [min_cost], color="#FF0060", s=100, zorder=5)
    ax.annotate(
        f"Min: {min_cost:.3f}",
        xy=(min_idx, min_cost),
        xytext=(min_idx + len(costs) * 0.05, min_cost),
        fontsize=10,
    )

    return ax


def plot_probabilities(probabilities: dict[str, float], top_k: int = 10, ax=None):
    """
    Plot bar chart of top-k solution probabilities.

    Args:
        probabilities: dict mapping bitstrings to probabilities
        top_k: number of top solutions to show
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Sort by probability and take top k
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    bitstrings, probs = zip(*sorted_probs) if sorted_probs else ([], [])

    colors = ["#FF0060" if i == 0 else "#0079FF" for i in range(len(bitstrings))]
    bars = ax.bar(range(len(bitstrings)), probs, color=colors)

    ax.set_xticks(range(len(bitstrings)))
    ax.set_xticklabels(bitstrings, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Bitstring (stem selection)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Solution Probability Distribution", fontsize=14)

    # Add probability labels on bars
    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob:.2%}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    return ax


def generate_final_structure_graph(
    rna: str, stems: List[Tuple[int, int, int]], ax=None
):
    """
    Generate network graph showing all selected stems in the final structure.
    """
    colors = {"G": "#FF0060", "A": "#F6FA70", "C": "#00DFA2", "U": "#0079FF"}
    n = len(rna)

    graph = nx.Graph()
    for i, base in enumerate(rna):
        graph.add_node(i, color=colors[base], label=base)
        if i + 1 < n:
            graph.add_edge(i, i + 1, style="backbone")

    # Add all stems
    for stem in stems:
        stem_span1 = list(range(stem[0] - 1, stem[0] + stem[2] - 1))
        stem_span2 = list(range(stem[1] - stem[2], stem[1]))
        stem_span2.reverse()
        for b1, b2 in zip(stem_span1, stem_span2):
            graph.add_edge(b1, b2, style="stem")

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))

    node_colors = [graph.nodes[node]["color"] for node in graph.nodes]
    node_labels = {node: graph.nodes[node]["label"] for node in graph.nodes}

    # Draw with different edge styles
    pos = nx.layout.kamada_kawai_layout(graph, scale=10)
    backbone_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("style") == "backbone"]
    stem_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("style") == "stem"]

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=300)
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=node_labels, font_size=12)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=backbone_edges, width=1, style="-")
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=stem_edges, width=2, style="--", edge_color="#FF0060"
    )

    ax.set_title(f"Predicted Structure ({len(stems)} stems)", fontsize=14)
    return ax
