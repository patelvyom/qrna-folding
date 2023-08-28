import matplotlib.pyplot as plt
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
