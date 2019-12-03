# Copyright (c) 2019 Simon van Heeringen <simon.vanheeringen@gmail.com>
#
# This module is free software. You can redistribute it and/or modify it under
# the terms of the MIT License, see the file LICENSE included with this
# distribution.
from functools import partial
import os
import re
import sys
from tempfile import NamedTemporaryFile
from multiprocessing import Pool
from pkg_resources import resource_filename

import xdg
import pandas as pd
import numpy as np
from fluff.fluffio import load_heatmap_data
from pybedtools import BedTool
from genomepy import Genome

CACHE_DIR = os.path.join(xdg.XDG_CACHE_HOME, "scepia")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def splitextgz(fname):
    """Return filename without .gz extension/

    Parameters
    ----------
    fname : str
        Filename.

    Returns
    -------
    str
        Filename without .gz extension.
    """
    if fname.endswith(".gz"):
        fname = os.path.splitext(fname)[0]
    return os.path.splitext(os.path.basename(fname))[0]


def count_bam(bedfile, bamfile, nthreads=4, window=2000):
    """Count reads in BAM file.

    Count reads from a BAM file in features of a BED file. An index file will
    be created if it doesn't exist.

    Parameters
    ----------
    bedfile : str
        Filename of BED file.

    bamfile : str
        Filename of BAM file.

    ncpus : int, optional
        Number of threads to use.

    window : int, optional
        Sizes of window from the center of regions in the BED file.

    Returns
    -------
    list
        List with counts.
    """
    f = partial(
        load_heatmap_data,
        datafile=bamfile,
        bins=1,
        up=window // 2,
        down=window // 2,
        rmdup=True,
        rpkm=False,
        rmrepeats=True,
        fragmentsize=None,
        dynam=False,
        guard=None,
    )

    df = pd.read_csv(bedfile, sep="\t", comment="#", header=None)
    df.iloc[:, 1] = df.iloc[:, 1].astype(int)
    df.iloc[:, 2] = df.iloc[:, 2].astype(int)
    total = df.shape[0]
    batch_size = total // nthreads + 1

    fnames = []
    for i in range(0, total, batch_size):
        tmp = NamedTemporaryFile(delete=False, suffix=".bed")
        df.iloc[i : i + batch_size].to_csv(
            tmp.name, sep="\t", index=False, header=False
        )
        fnames.append(tmp.name)

    pool = Pool(nthreads)

    total = []
    for counts in pool.map(f, fnames):
        total += [x[0] for x in counts[2]]

    return total


def quantile_norm(x, target=None):
    """Quantile normalize a 2D array.

    Parameters
    ----------
    x : numpy.array
        a 2D numpy array
    target : numpy.array, optional
        Reference distribution to use for normalization.
        If not supplied, the mean of x is used.

    Returns
    -------
    numpy.array
        Normalized array.
    """

    def quantile(x, y):
        return y[x.argsort().argsort()]

    if target is None:
        sidx = x.argsort(axis=0)
        target = x[sidx, np.arange(sidx.shape[1])].mean(1)
    func = partial(quantile, y=target)
    return np.apply_along_axis(func, 0, x)


def weigh_distance(dist):
    """Return enhancer weight based upon distance.

    Parameters
    ----------
    dist : float
        Genomic distance.

    Returns
    -------
    float based on distance.
    """
    mu = -np.log(1 / 3) / 10000
    d = np.abs(dist) - 5000
    d[d < 0] = 0
    w = 2 * np.exp(-(mu * d)) / (1 + np.exp(-(mu * d)))
    return w


def create_link_file(meanstd_file, genes_file, genome="hg38"):
    # Read enhancer locations
    if meanstd_file.endswith("feather"):
        tmp = pd.read_feather(meanstd_file)["index"]
    else:
        tmp = pd.read_csv(meanstd_file, sep="\t")["index"]
    enhancers = BedTool.from_dataframe(tmp.str.split("[-:]", expand=True))

    # Calculating overlap with certain distance
    g = Genome(genome).props["sizes"]["sizes"]
    genes = BedTool(genes_file).slop(b=100000, g=g).cut([0, 1, 2, 3])
    overlap = genes.intersect(b=enhancers, wo=True)
    overlap = overlap.to_dataframe().iloc[:, 3:7]
    overlap.columns = ["gene", "chrom", "start", "end"]
    overlap["loc"] = (
        overlap["chrom"]
        + ":"
        + overlap["start"].astype(str)
        + "-"
        + overlap["end"].astype(str)
    )
    overlap["pos"] = ((overlap["start"] + overlap["end"]) / 2).astype(int)
    overlap = overlap[["gene", "loc", "pos"]]
    return overlap


def link_it_up(
    outfile, signal, meanstd_file=None, genes_file=None, names_file=None, threshold=2.0
):
    """Return file with H3K27ac "score" per gene.

    H3K27ac signal is summarized per gene weighted by distance.

    Parameters
    ----------
    outfile : str
        Name of output file.
    signal : panda.DataFrame
        DataFrame with index chrom:start-end and a column named 'signal'.
    meanstd_file : str, optional
        Name of file with the mean and standard deviation of the signal per enhancer.
    genes_file : str, optional
        Name of gene annotation in BED format.
    names_file : str, optional
        Name of the file linking gene identifiers to gene names.
    threshold : float, optional
        Only use enhancers with at least a signal above threshold. Default is 2.0.
    """
    if meanstd_file is None:
        meanstd_file = resource_filename(__name__, "data/remap.hg38.meanstd.tsv.gz")
    if genes_file is None:
        genes_file = resource_filename(
            __name__, "data/gencode.v30.TSS.all_transcripts.merged1kb.bed.gz"
        )
    if names_file is None:
        names_file = resource_filename(__name__, "data/ens2name.txt")

    ens2name = pd.read_csv(names_file, sep="\t", index_col=0, names=["gene", "name"])

    link_file = os.path.join(
        CACHE_DIR,
        ".".join((splitextgz(meanstd_file), splitextgz(genes_file), "feather")),
    )

    if not os.path.exists(link_file):
        genome = re.sub(
            r"[^\.]+\.(.*)\.meanstd.*", "\\1", os.path.basename(meanstd_file)
        )
        sys.stdout.write("Creating link file with genome {}\n".format(genome))
        link = create_link_file(meanstd_file, genes_file, genome=genome)
        sys.stdout.write(f"Saving to {CACHE_DIR}\n")
        link.to_feather(link_file)
    else:
        # Read enhancer to gene links
        link = pd.read_feather(link_file)

    # Read genes
    genes = pd.read_csv(
        genes_file,
        sep="\t",
        index_col=2,
        usecols=[0, 1, 3, 5],
        names=["chrom", "start", "gene", "strand"],
    )

    link = link.join(
        signal[signal["signal"] >= threshold][["signal"]], on="loc"
    ).dropna()
    link = link.dropna()

    # Split multiple genes
    link["gene"] = link["gene"].str.split(",")
    lst_col = "gene"
    link = pd.DataFrame(
        {
            col: np.repeat(link[col].values, link[lst_col].str.len())
            for col in link.columns.drop(lst_col)
        }
    ).assign(**{lst_col: np.concatenate(link[lst_col].values)})[link.columns]
    link = link.join(genes, on="gene")

    # Distance weight
    link["dist"] = link["start"] - link["pos"]
    link["dist_weight"] = weigh_distance(link["dist"])

    link["contrib"] = link["dist_weight"] * link["signal"]
    link = link.sort_values("contrib", ascending=False)[["gene", "loc", "contrib"]]
    link = (
        link.groupby(["gene", "loc"])
        .first()
        .reset_index()
        .groupby("gene")
        .sum()[["contrib"]]
    )
    link = link.join(ens2name).dropna().set_index("name")
    sys.stderr.write(f"Writing output file {outfile}\n")
    link.to_csv(outfile, sep="\t")


def generate_signal(bam_file, window, meanstd_file=None, target_file=None, nthreads=4):
    """Read BAM file and return normalized read counts.

    Read counts are determined in specified window, log-transformed and quantile-normalized.

    Parameters
    ----------
    bam_file : str
        Name of BAM file.
    window : int
        Size of window to use.
    meanstd_file : str, optional
        Name of reference file with regions (chrom:start-end) in first column, mean
        in second column and standard deviation in third column.
    target_file : str, optional
        Name of file containing reference values for quantile normalization (npz format)
    nthread : int, optional
        Number of threads to use, default is 4.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing normalized signal
    """
    if meanstd_file is None:
        meanstd_file = resource_filename(__name__, "data/remap.hg38.meanstd.tsv.gz")
    if target_file is None:
        target_file = resource_filename(__name__, "data/remap.hg38.target.npz")

    with NamedTemporaryFile(prefix=f"scepia.", suffix=".bed") as f:
        if meanstd_file.endswith("feather"):
            meanstd = pd.read_feather(meanstd_file)
        else:
            meanstd = pd.read_csv(meanstd_file, sep="\t")
        meanstd["index"].str.split("[:-]", expand=True).to_csv(
            f.name, sep="\t", index=False, header=False
        )
        result = count_bam(f.name, bam_file, nthreads=nthreads, window=window)
        meanstd["signal"] = result

    # Normalization
    sys.stderr.write("Normalizing\n")
    meanstd["signal"] = np.log1p(meanstd["signal"])
    target = np.load(target_file)["target"]
    # np.random.shuffle(target)
    meanstd["signal"] = quantile_norm(meanstd["signal"].values, target)
    meanstd["signal"] = (meanstd["signal"] - meanstd["mean"]) / meanstd["std"]
    return meanstd.set_index("index")[["signal"]]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
