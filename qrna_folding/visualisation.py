from typing import Tuple

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
