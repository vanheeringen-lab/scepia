# Copyright (c) 2019 Simon van Heeringen <simon.vanheeringen@gmail.com>
#
# This module is free software. You can redistribute it and/or modify it under
# the terms of the MIT License, see the file LICENSE included with this
# distribution.
# Typing
from typing import Optional, Tuple

from adjustText import adjust_text
from anndata import AnnData

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec
import seaborn as sns

from gimmemotifs.motif import read_motifs


def plot_volcano_corr(
    adata: AnnData,
    max_pval: Optional[float] = 0.05,
    n_anno: Optional[int] = 40,
    size_anno: Optional[float] = 7,
    palette: Optional[str] = None,
    alpha: Optional[float] = 0.8,
    linewidth: Optional[float] = 0,
    sizes: Optional[Tuple[int, int]] = (3, 20),
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    sns.set_style("ticks")

    plot_data = (
        adata.uns["scepia"]["correlation"]
        .reset_index()
        .sort_values("p_adj")
        .groupby("factor")
        .first()
    )

    g = sns.scatterplot(
        data=plot_data,
        y="-log10(p-value)",
        x="correlation",
        size="motif_stddev",
        hue="motif_stddev",
        palette=palette,
        sizes=sizes,
        linewidth=linewidth,
        alpha=alpha,
        ax=ax,
        **kwargs,
    )
    g.legend_.remove()

    factors = (
        plot_data[(plot_data["p_adj"] <= max_pval)].sort_values("p_adj").index[:n_anno]
    )
    x = plot_data.loc[factors, "correlation"]
    y = plot_data.loc[factors, "-log10(p-value)"]

    texts = []
    for s, xt, yt in zip(factors, x, y):
        texts.append(plt.text(xt, yt, s, {"size": size_anno}))

    adjust_text(
        texts, arrowprops=dict(arrowstyle="-", color="black"),
    )
    # plt.xlabel("Correlation (motif vs. factor expression)")
    # plt.ylabel("Significance (-log10 p-value)")
    return g


def plot(
    adata: AnnData,
    max_pval: Optional[float] = 0.05,
    n_anno: Optional[int] = 40,
    size_anno: Optional[float] = 7,
    palette: Optional[str] = None,
    alpha: Optional[float] = 0.8,
    linewidth: Optional[float] = 0,
    sizes: Optional[Tuple[int, int]] = (3, 20),
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:

    motifs = read_motifs(adata.uns["scepia"]["pfm"], as_dict=True)
    n_motifs = 8

    fig = plt.figure(figsize=(5, n_motifs * 0.75))
    gs = gridspec.GridSpec(n_motifs, 5)

    ax = fig.add_subplot(gs[:, :4])
    plot_volcano_corr(adata, ax=ax, size_anno=8)

    factors = (
        adata.uns["scepia"]["correlation"]
        .groupby("factor")
        .min()
        .sort_values("p_adj")
        .index[:n_motifs]
    )

    for i in range(n_motifs):
        factor = factors[i]
        motif = (
            adata.uns["scepia"]["correlation"]
            .loc[factor]
            .sort_values("p_adj")
            .iloc[0]
            .motif
        )
        ax = fig.add_subplot(gs[i, 4:])
        motifs[motif].plot_logo(ax=ax, ylabel=False, title=False)
        plt.title(factor)
        ax.title.set_fontsize(8)
        ax.axis("off")

    plt.tight_layout()
    return fig
