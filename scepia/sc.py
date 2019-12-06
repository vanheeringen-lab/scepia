# Copyright (c) 2019 Simon van Heeringen <simon.vanheeringen@gmail.com>
#
# This module is free software. You can redistribute it and/or modify it under
# the terms of the MIT License, see the file LICENSE included with this
# distribution.
from collections import Counter
import inspect
from multiprocessing import Pool
import os
import shutil
import sys
import tarfile
from tempfile import NamedTemporaryFile
import urllib.request

# Typing
from typing import List, Optional, Tuple

from adjustText import adjust_text
from anndata import AnnData
from appdirs import user_cache_dir
from gimmemotifs.motif import read_motifs
from gimmemotifs.moap import moap
from gimmemotifs.maelstrom import run_maelstrom
from gimmemotifs.rank import rankagg
from gimmemotifs.utils import pfmfile_location
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import scanpy.api as sc
import seaborn as sns
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import (  # noqa: F401
    BayesianRidge,
    LogisticRegression,
    LassoCV,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from scipy.sparse import issparse
from scipy.stats import pearsonr, percentileofscore
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
from yaml import load


CACHE_DIR = os.path.join(user_cache_dir("scepia"))
expression = None
f_and_m = None
_corr_adata = None


class MotifAnnData(AnnData):
    """Extended AnnData class.

    Add the functionality to correctly save and load an AnnData object with
    motif annotation results.
    """

    df_keys = ["motif_activity", "factor2motif", "correlation"]

    def __init__(self, adata):
        super().__init__(adata)

    def _remove_additional_data(self) -> None:
        # Motif columns need to be removed, as the hdf5 backend cannot
        # store the data otherwise (header too long).
        self.obs = self.obs.drop(columns=self.uns["scepia"]["motif_activity"].index)
        # DataFrames are not supported in the h5ad format. By converting them
        # dictionaries the can be restored to DataFrame format after loading.
        for k in self.df_keys:
            self.uns["scepia"][k] = self.uns["scepia"][k].to_dict()

    def _restore_additional_data(self) -> None:
        # In this case it works for an AnnData object that contains no
        # additional motif information
        if "motif" not in self.uns:
            return

        # Restore all DataFrames in uns
        for k in self.df_keys:
            self.uns["scepia"][k] = pd.DataFrame(self.uns["scepia"][k])

        # Make sure the cell types are in the correct order
        self.uns["scepia"]["motif_activity"] = self.uns["scepia"]["motif_activity"][
            self.uns["scepia"]["cell_types"]
        ]
        #  The cell type-specific motif activity needs to be recreated.
        cell_motif_activity = pd.DataFrame(
            self.uns["scepia"]["motif_activity"] @ self.obsm["X_cell_types"].T
        ).T
        cell_motif_activity.index = self.obs_names
        self.obs = self.obs.drop(
            columns=cell_motif_activity.columns.intersection(self.obs.columns)
        )
        self.obs = self.obs.join(cell_motif_activity)

    def write(self, *args, **kwargs) -> None:
        """Write a MotifAnnData object.

        All DataFrames in uns are converted to dictionaries and motif columns
        are removed from obs.
        """
        self._remove_additional_data()
        super().write(*args, **kwargs)
        # If we don't restore it, the motif annotation will be useless
        # after saving a MotifAnnData object.
        self._restore_additional_data()


def read(filename: str) -> AnnData:
    """Read a MotifAnnData object from a h5ad file.

    Parameters
    ----------
    filename : `str`
        Name of a h5ad file.

    Return
    ------
    MotifAnnData object.
    """
    logger.info("reading .h5ad file")
    adata = MotifAnnData(sc.read(filename))
    logger.info("done")
    logger.info("converting and populating scepia motif properties")
    adata._restore_additional_data()
    logger.info("done")
    return adata


def motif_mapping(
    pfm: Optional[str] = None,
    genes: Optional[List[str]] = None,
    indirect: Optional[bool] = True,
) -> pd.DataFrame:
    """Read motif annotation and return as DataFrame.

    Parameters
    ----------
    pfm : `str`, optional
        Name of pfm file. Should have an associated file with mapping from
        motif to factors, with the .motif2factors.txt extension.
    genes : `list`, optional
        List of gene names to include. If None all genes will be included.
    indirect : `boolean`, optional
        Include indirect factors in the annotation. Default True. If set to
        False only factors for which there is direct evidence will be
        used.

    Returns
    -------
    `pd.DataFrame`
        DataFrame with motif names as index and an associated column with TFs
        that bind to the motifs.
    """
    m = read_motifs(pfm)

    m2f = {}
    for motif in m:
        factors = motif.factors["direct"]

        # Also include factors for which there is no direct evidence
        if indirect:
            factors += motif.factors.get("indirect\nor predicted", [])

        # Create a string of comma-separated factors per motif
        factors = list(set([factor.upper() for factor in factors]))
        for factor in factors:
            if genes is None or factor in genes:
                if motif.id not in m2f:
                    m2f[motif.id] = factor
                else:
                    m2f[motif.id] = m2f[motif.id] + "," + factor

    m2f = pd.DataFrame({"factors": m2f})
    return m2f


def read_enhancer_data(
    fname: str,
    anno_fname: Optional[str] = None,
    anno_from: Optional[str] = None,
    anno_to: Optional[str] = None,
    scale: Optional[bool] = False,
) -> pd.DataFrame:
    if fname.endswith(".txt"):
        df = pd.read_csv(fname, sep="\t", comment="#", index_col=0)
    elif fname.endswith(".csv"):
        df = pd.read_csv(fname, comment="#", index_col=0)
    elif fname.endswith("feather"):
        df = pd.read_feather(fname)
        df = df.set_index(df.columns[0])

    # Remove mitochondrial regions
    df = df.loc[~df.index.str.contains("chrM")]

    # Map column names using annotation file
    if anno_fname:
        md = pd.read_csv(anno_fname, sep="\t", comment="#")
        if anno_from is None:
            anno_from = md.columns[0]
        if anno_to is None:
            anno_to = md.columns[1]
        md = md.set_index(anno_from)[anno_to]
        df = df.rename(columns=md)
        # Merge identical columns
        df = df.groupby(df.columns, axis=1).mean()
    return df


def annotate_with_k27(
    adata: AnnData,
    gene_df: pd.DataFrame,
    n_neighbors: Optional[int] = 20,
    center_expression: Optional[bool] = True,
    model: Optional[str] = "BayesianRidge",
    use_neighbors: Optional[bool] = True,
    use_raw: Optional[bool] = False,
    subsample: Optional[bool] = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate single cell data.
    """
    # Only use genes that overlap
    common_genes = adata.var_names.intersection(gene_df.index).unique()
    # print(common_genes)
    # Create expression DataFrame based on common genes
    if use_raw:
        expression = pd.DataFrame(
            np.log1p(adata.raw.X[:, adata.var_names.isin(gene_df.index)].todense()),
            index=adata.obs_names,
            columns=common_genes,
        ).T
    else:
        expression = pd.DataFrame(
            adata.X[:, adata.var_names.isin(gene_df.index)],
            index=adata.obs_names,
            columns=common_genes,
        ).T

    if center_expression:
        expression = expression.sub(expression.mean(1), 0)

    # Get sampled idxs
    N = 100000
    unique_cell_types = adata.obs["louvain"].unique()
    counts = adata.obs.groupby("louvain").count().iloc[:, 0].to_dict()
    ids = np.arange(adata.shape[0])
    idxs = []
    for cell_type in unique_cell_types:
        if counts[cell_type] <= N:
            idx = ids[adata.obs["louvain"] == cell_type]
        else:
            idx = np.random.choice(
                ids[adata.obs["louvain"] == cell_type], N, replace=False
            )
        idxs.extend(idx)

    X = gene_df.loc[common_genes]
    model = getattr(sys.modules[__name__], model)()
    kf = KFold(n_splits=5)

    result = []
    df_coef = pd.DataFrame(index=gene_df.columns)
    with tqdm(total=len(idxs), file=sys.stdout) as pbar:
        for i in idxs:
            if use_neighbors:
                my_neighbors = (
                    pd.DataFrame(
                        (adata.uns["neighbors"]["connectivities"][i] != 0).todense()
                    )
                    .iloc[0]
                    .values
                )
                y = expression.loc[:, my_neighbors].mean(1)
            else:
                y = expression.iloc[:, i]

            if subsample:
                cts = []
                for _, idx in kf.split(X):
                    model.fit(X.iloc[idx], y[idx])
                    coef = pd.DataFrame(model.coef_, index=gene_df.columns)
                    ct = coef.sort_values(0).tail(1).index[0]
                    cts.append(ct)
                # df_coef[i] = 0
                top_ct = Counter(cts).most_common()[0][0]
                df_coef[i] = pd.DataFrame.from_dict(Counter(cts), orient="index")
                df_coef[i] = df_coef[i].fillna(0)
            else:
                model.fit(X, y)
                if model == "LogisticRegression":
                    coef = pd.DataFrame(model.coef_[0], index=gene_df.columns)
                else:
                    coef = pd.DataFrame(model.coef_, index=gene_df.columns)
                df_coef[i] = coef[0]
                top_ct = coef.sort_values(0).tail(1).index[0]
            # print("{}\t{}".format(top_ct, adata.obs['cell_type'].iloc[i]), coef.sort_values(0).tail(5).index)
            result.append([top_ct])
            pbar.update(1)

    df_coef = df_coef[sorted(df_coef.columns)]
    return (
        pd.DataFrame(result, columns=["cell_annotation"], index=adata.obs_names[idxs]),
        df_coef,
    )


def relevant_cell_types(
    adata: AnnData,
    gene_df: pd.DataFrame,
    n_top_genes: Optional[int] = 1000,
    cv: Optional[int] = 5,
) -> List[str]:
    """Select relevant cell types for annotation and motif inference.

    Based on Multitask Lasso regression a subset of features (cell type
    profile) will be selected. Expression is averaged over louvain clusters
    and selected features are forced to be the same over all clusters.
    Requires louvain clustering to be run on the `adata` object.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    gene_df : :class:`pandas.DataFrame`
        Gene-based reference data.
    n_top_genes : `int`, optional (default: 1000)
        Number of variable genes is used. If `n_top_genes` is greater than the
        number of hypervariable genes in `adata` then all variable genes are
        used.
    cv : `int`, optional (default: 5)
        Folds for cross-validation.

    Returns
    -------
    `list`
        Cell types ordered by the mean absolute coefficient over clusters in
        descending order.
    """
    logger.info("selecting reference cell types")
    common_genes = list(gene_df.index[gene_df.index.isin(adata.var_names)])
    expression = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names).T
    expression = expression.loc[common_genes]
    expression.columns = adata.obs["louvain"]
    expression = expression.groupby(expression.columns, axis=1).mean()

    var_genes = (
        adata.var.loc[common_genes, "dispersions_norm"]
        .sort_values()
        .tail(n_top_genes)
        .index
    )
    expression = expression.loc[var_genes]
    X = gene_df.loc[var_genes]
    g = MultiTaskLassoCV(cv=cv, n_jobs=24, selection="random")
    g.fit(X, expression)
    bla = (
        pd.DataFrame(g.coef_, index=expression.columns, columns=X.columns)
        .sum(0)
        .sort_values()
    )
    bla = (
        np.abs(pd.DataFrame(g.coef_, index=expression.columns, columns=X.columns))
        .sum(0)
        .sort_values(ascending=False)
    )
    cell_types = bla[bla != 0].index
    logger.info("{} out of {} selected".format(len(cell_types), gene_df.shape[1]))
    logger.info("top 5:")
    for cell_type in cell_types[:5]:
        logger.info(f" * {cell_type}")
    return list(cell_types)


def validate_adata(adata: AnnData) -> None:
    """Check if adata contains the necessary prerequisites to run the
    motif inference.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    """
    if adata.raw is None:
        raise ValueError("Please save the raw expression data in the .raw property.")

    if "neighbors" not in adata.uns or "louvain" not in adata.obs:
        raise ValueError("Please run louvain clustering first.")


def load_reference_data(
    config: dict, data_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("loading reference data")
    fname_enhancers = os.path.join(data_dir, config["enhancers"])
    fname_genes = os.path.join(data_dir, config["genes"])
    anno_fname = config.get("anno_file")
    if anno_fname is not None:
        anno_fname = os.path.join(data_dir, anno_fname)
    anno_to = config.get("anno_to")
    anno_from = config.get("anno_from")
    # H3K27ac signal in enhancers
    enhancer_df = read_enhancer_data(
        fname_enhancers, anno_fname=anno_fname, anno_to=anno_to
    )

    # H3K27ac signal summarized per gene
    gene_df = read_enhancer_data(
        fname_genes, anno_fname=anno_fname, anno_to=anno_to, anno_from=anno_from
    )
    return enhancer_df, gene_df


def change_region_size(series: pd.Series, size: Optional[int] = 200) -> pd.Series:
    if not isinstance(series, pd.Series):
        if hasattr(series, "to_series"):
            series = series.to_series()
        else:
            series = pd.Series(series)

    loc = series.str.split("[:-]", expand=True)
    loc["start"] = (loc[1].astype(int) + loc[2].astype(int)) // 2 - (size // 2)
    loc["end"] = (loc["start"] + size).astype(str)
    loc["start"] = loc["start"].astype("str")
    return loc[0] + ":" + loc["start"] + "-" + loc["end"]


def infer_motifs(
    adata: AnnData,
    dataset: str,
    pfm: Optional[str] = None,
    min_annotated: Optional[int] = 50,
    num_enhancers: Optional[int] = 10000,
    maelstrom: Optional[bool] = False,
) -> None:
    """Infer motif ativity for single cell RNA-seq data.

    The adata object is modified with the following fields.

    **X_cell_types** : `adata.obsm` field
        Cell type coefficients.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    dataset : `str`
        Name of reference data set or directory with reference data.
    pfm : `str`, optional (default: None)
        Name of motif file in PFM format. The GimmeMotifs default is used
        if this parameter is not specified. This can be a filename, or a
        pfm name support by GimmeMotifs such as `JASPAR2018_vertebrates`.
        If a custom PFM file is specified, there should also be an associated
        `.motif2factors.txt` file.
    min_annotated : `int`, optional (default: 50)
        Cells that are annotated with cell types less than this number will be
        annotated as "other".
    num_enhancers : `int`, optional (default: 10000)
        Number of enhancers to use for motif activity analysis.
    maelstrom : `boolean`, optional (default: False)
        Use maelstrom instead of ridge regression for motif activity analysis.
    """

    use_name = True

    validate_adata(adata)

    data_dir = locate_data(dataset)

    with open(os.path.join(data_dir, "info.yaml")) as f:
        config = load(f)

    logger.debug(config)
    adata.uns["scepia"] = {}
    link_file = os.path.join(data_dir, config.get("link_file"))

    gene_map_file = config.get("gene_mapping")
    if gene_map_file is not None:
        gene_map_file = os.path.join(data_dir, gene_map_file)

    enhancer_df, gene_df = load_reference_data(config, data_dir)

    # Determine relevant reference cell types.
    # All other cell types will not be used for motif activity and
    # cell type annotation.
    cell_types = relevant_cell_types(adata, gene_df)
    adata.uns["scepia"]["cell_types"] = cell_types

    logger.info("linking variable genes to differential enhancers")
    link = pd.read_feather(link_file)
    if use_name:
        ens2name = pd.read_csv(
            gene_map_file, sep="\t", index_col=0, names=["identifier", "name"]
        )
        link = link.join(ens2name, on="gene").dropna()
        link = link.set_index("name")

    enh_genes = adata.var_names[adata.var_names.isin(link.index)]
    var_enhancers = change_region_size(link.loc[enh_genes, "loc"]).unique()

    enhancer_df.index = change_region_size(enhancer_df.index)
    enhancer_df = enhancer_df.loc[var_enhancers, cell_types]
    enhancer_df = enhancer_df.groupby(enhancer_df.columns, axis=1).mean()
    enhancer_df.loc[:, :] = scale(enhancer_df)
    # Select top most variable enhancers
    enhancer_df = enhancer_df.loc[
        enhancer_df.var(1).sort_values().tail(num_enhancers).index
    ]
    # Center by mean
    enhancer_df = enhancer_df.sub(enhancer_df.mean(1), axis=0)
    fname = NamedTemporaryFile(delete=False).name
    logger.debug(f"enhancer filename: {fname}")
    enhancer_df.to_csv(fname, sep="\t")
    logger.info("inferring motif activity")

    pfm = pfmfile_location(pfm)
    adata.uns["scepia"]["pfm"] = pfm

    if maelstrom:
        run_maelstrom(
            fname,
            "hg38",
            "tmp.lala",
            methods=["bayesianridge", "lightningregressor", "xgboost"],
        )

        motif_act = pd.read_csv(
            os.path.join("tmp.lala", "final.out.csv"),
            sep="\t",
            comment="#",
            index_col=0,
        )
    else:
        motif_act = moap(
            fname,
            scoring="score",
            genome="hg38",
            method="bayesianridge",
            pfmfile=pfm,
            ncpus=12,
        )

    adata.uns["scepia"]["motif_activity"] = motif_act[adata.uns["scepia"]["cell_types"]]
    logger.info("annotating cells")
    annotation_result, df_coef = annotate_with_k27(
        adata,
        gene_df[cell_types],
        use_neighbors=True,
        model="BayesianRidge",
        subsample=False,
        use_raw=False,
    )
    adata.obsm["X_cell_types"] = df_coef.T[adata.uns["scepia"]["cell_types"]].values

    #    adata.obs = adata.obs.drop(
    #        columns=annotation_result.columns.intersection(adata.obs.columns)
    #    )
    #    adata.obs = adata.obs.join(annotation_result)
    #
    #    count = adata.obs.groupby("cell_annotation").count().iloc[:, 0]
    #    valid = count[count > min_annotated].index
    #    adata.obs["cell_annotation"] = adata.obs["cell_annotation"].astype(str)
    #    adata.obs.loc[
    #        ~adata.obs["cell_annotation"].isin(valid), "cell_annotation"
    #    ] = "other"

    assign_cell_types(adata, min_annotated=1)
    cluster_count = (
        adata.obs.groupby(["louvain", "cell_annotation"]).count().iloc[:, 0].dropna()
    )
    cluster_anno = (
        cluster_count.sort_values()
        .reset_index()
        .drop_duplicates(subset="louvain", keep="last")
        .set_index("louvain")
    )

    cluster_anno = cluster_anno[["cell_annotation"]].rename(
        columns={"cell_annotation": "cluster_annotation"}
    )
    adata.obs = adata.obs.join(cluster_anno, on="louvain")
    assign_cell_types(adata)

    logger.info("calculating cell-specific motif activity")
    cell_motif_activity = (
        adata.uns["scepia"]["motif_activity"] @ adata.obsm["X_cell_types"].T
    ).T
    cell_motif_activity.index = adata.obs_names
    adata.obs = adata.obs.drop(
        columns=cell_motif_activity.columns.intersection(adata.obs.columns)
    )
    adata.obs = adata.obs.join(cell_motif_activity)

    correlate_tf_motifs(adata)

    add_activity(adata)

    return MotifAnnData(adata)


def correlate_tf_motifs(adata: AnnData) -> None:
    """Correlate inferred motif activity with TF expression.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    """
    logger.info("correlating TFs with motifs")
    m2f = motif_mapping(adata.uns["scepia"]["pfm"])
    if issparse(adata.raw.X):
        expression = pd.DataFrame(
            adata.raw.X.todense(), index=adata.obs_names, columns=adata.raw.var_names
        ).T
    else:
        expression = pd.DataFrame(
            adata.raw.X, index=adata.obs_names, columns=adata.raw.var_names
        ).T

    cor = []
    for motif in tqdm(adata.uns["scepia"]["motif_activity"].index):
        if motif in m2f.index:
            for factor in m2f.loc[motif, "factors"].split(","):
                if factor in expression.index:
                    cor.append(
                        (
                            motif,
                            factor,
                            *pearsonr(adata.obs[motif], expression.loc[factor]),
                        )
                    )
    cor = pd.DataFrame(cor, columns=["motif", "factor", "corr", "pval"])

    if cor.shape[0] == 0:
        logger.warn("no factor annotation for motifs found")
        return

    cor["padj"] = multipletests(cor["pval"], method="fdr_bh")[1]
    cor["abscorr"] = np.abs(cor["corr"])

    # Determine putative roles of TF based on sign of correlation coefficient.
    # This will not always be correct, as many factors have a high correlation
    # with more than one motif, both positive and negative. In that case it's
    # hard to assign a role.
    cor["putative_role"] = "activator"
    cor.loc[(cor["corr"] + cor["abscorr"]) < 1e-6, "putative_role"] = "repressor"
    adata.uns["scepia"]["factor2motif"] = cor


def add_activity(adata: AnnData):
    """Get factor activity"""
    if "scepia" not in adata.uns:
        raise ValueError(
            "Could not find motif information. Did you run infer_motifs() first?"
        )
    if "factor2motif" not in adata.uns["scepia"]:
        logger.warn("Cannot determine factor activity without factor annotation")
        return

    gm = GaussianMixture(n_components=2, covariance_type="full")
    f2m = adata.uns["scepia"]["factor2motif"]
    for factor in f2m["factor"].unique():
        motif = (
            f2m[f2m["factor"] == factor].sort_values("pval").head(1)["motif"].values[0]
        )
        gm.fit(adata.obs[motif].values.reshape(-1, 1) * 10)
        adata.obs[f"{factor}_activity"] = gm.predict_proba(adata.obs[[motif]] * 10)[
            :, gm.means_.argmax()
        ]


def assign_cell_types(adata: AnnData, min_annotated: Optional[int] = 50) -> None:
    adata.obs["cell_annotation"] = (
        pd.Series(adata.uns["scepia"]["cell_types"])
        .iloc[adata.obsm["X_cell_types"].argmax(1)]
        .values
    )
    count = adata.obs.groupby("cell_annotation").count().iloc[:, 0]
    valid = count[count > min_annotated].index
    adata.obs["cell_annotation"] = adata.obs["cell_annotation"].astype(str)
    adata.obs.loc[
        ~adata.obs["cell_annotation"].isin(valid), "cell_annotation"
    ] = "other"


def locate_data(dataset: str, version: Optional[float] = None) -> str:
    """Locate reference data for cell type annotation.

    Data set will be downloaded if it can not be found.

    Parameters
    ----------
    dataset : `str`
        Name of dataset. Can be a local directory.
    version : `float`, optional
        Version of dataset

    Returns
    -------
    `str`
        Absolute path to data set directory.
    """
    for datadir in [os.path.expanduser(dataset), os.path.join(CACHE_DIR, dataset)]:
        if os.path.isdir(datadir):
            if os.path.exists(os.path.join(datadir, "info.yaml")):
                return datadir
            else:
                raise ValueError(f"info.yaml not found in directory {datadir}")

    # Data file can not be found
    # Read the data directory from the installed module
    # Once the github is public, it can be read from the github repo directly
    install_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    df = pd.read_csv(os.path.join(install_dir, "../data/data_directory.txt"), sep="\t")
    df = df[df["name"] == dataset]
    if df.shape[0] > 0:
        if version is None:
            url = df.sort_values("version").tail(1)["url"].values[0]
        else:
            url_df = df.loc[df["version"] == version]
            if url_df.shape[0] > 0:
                url = url_df["url"].values[0]
            else:
                raise ValueError(f"Dataset {dataset} with version {version} not found.")
        datadir = os.path.join(CACHE_DIR, dataset)
        os.mkdir(datadir)
        datafile = os.path.join(datadir, os.path.split(url)[-1])
        logger.info(f"Downloading {dataset} data files to {datadir}...\n")
        with urllib.request.urlopen(url) as response, open(datafile, "wb") as outfile:
            shutil.copyfileobj(response, outfile)

        logger.info("Extracting files...\n")
        tf = tarfile.open(datafile)
        tf.extractall(datadir)
        os.unlink(datafile)

        return datadir
    else:
        raise ValueError(f"Dataset {dataset} not found.")


def _run_correlation(args: Tuple[int, int, bool]) -> pd.DataFrame:
    """Calculate correlation between motif activity and factor expression.
    """
    seed, it, do_shuffle = args
    global expression
    global f_and_m
    global _corr_adata

    if do_shuffle:
        shape = _corr_adata.uns["scepia"]["motif_activity"].shape
        motif_activity = shuffle(
            _corr_adata.uns["scepia"]["motif_activity"].values.flatten(),
            random_state=seed,
        ).reshape(shape[1], shape[0])

    else:
        motif_activity = _corr_adata.uns["scepia"]["motif_activity"].T.values
    cell_motif_activity = pd.DataFrame(
        _corr_adata.obsm["X_cell_types"] @ motif_activity
    )
    cell_motif_activity.columns = _corr_adata.uns["scepia"]["motif_activity"].index

    correlation = []
    for factor, motifs in f_and_m.items():
        correlation.append(
            (
                factor,
                pearsonr(cell_motif_activity[motifs].mean(1), expression.loc[factor])[
                    0
                ],
            )
        )
    correlation = pd.DataFrame(correlation, columns=["factor", f"corr"]).set_index(
        "factor"
    )

    return correlation


def determine_significance(
    adata: AnnData,
    n_rounds: Optional[int] = 10000,
    ncpus: Optional[int] = 12,
    corr_quantile: Optional[float] = 0.5,
) -> None:
    """Determine significance of motif-TF correlations by Monte Carlo simulation.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    n_rounds : `int`, optional
        Number of Monte Carlo simulations. The default is 10000.
    ncpus : `int`, optional
        Number of threads to use.
    """
    # Create DataFrame of gene expression from the raw data in the adata object.
    # We use the raw data as it will contain many more genes. Relevant transcription
    # factors are not necessarily called as hyper-variable genes.
    if "scepia" not in adata.uns:
        raise ValueError(
            "Could not find motif information. Did you run infer_motifs() first?"
        )

    global expression
    global f_and_m
    global _corr_adata
    _corr_adata = adata
    if issparse(adata.raw.X):
        expression = pd.DataFrame(
            adata.raw.X.todense(), index=adata.obs_names, columns=adata.raw.var_names
        ).T
    else:
        expression = pd.DataFrame(
            adata.raw.X, index=adata.obs_names, columns=adata.raw.var_names
        ).T

    m_and_f = []
    f_and_m = {}
    m2f = motif_mapping(indirect=False)
    for motif in adata.uns["scepia"]["motif_activity"].index:
        if motif in m2f.index:
            factors = [
                f for f in m2f.loc[motif, "factors"].split(",") if f in expression.index
            ]
            m_and_f.append([motif, factors])

            for factor in factors:
                if factor in f_and_m:
                    f_and_m[factor].append(motif)
                else:
                    f_and_m[factor] = [motif]

    correlation = pd.DataFrame(index=f_and_m.keys())
    correlation = correlation.join(_run_correlation((0, 0, False)))
    correlation.columns = ["actual_corr"]
    correlation["abs.actual_corr"] = np.abs(correlation["actual_corr"])

    std_lst = []
    for m, fs in m_and_f:
        for f in fs:
            std_lst.append([m, f])
    std_df = pd.DataFrame(std_lst, columns=["motif", "factor"]).set_index("motif")
    std_df = (
        pd.DataFrame(adata.uns["scepia"]["motif_activity"].std(1))
        .join(std_df)
        .groupby("factor")
        .max()
        .sort_values(0)
    )
    std_df.columns = ["std"]
    if "std" in correlation.columns:
        correlation = correlation.drop(columns=["std"])
    correlation = correlation.join(std_df)

    min_corr = correlation["abs.actual_corr"].quantile(corr_quantile)
    for factor in correlation[correlation["abs.actual_corr"] < min_corr].index:
        del f_and_m[factor]

    tmp_corr = correlation[correlation["abs.actual_corr"] >= min_corr]
    # Run n_rounds of correlation with shuffled motif activities.
    # The last iteration will be with the real motif activities.
    n_rounds = n_rounds
    logger.info("running Monte Carlo")
    pool = Pool(ncpus)
    for i, corr_iter, in tqdm(
        enumerate(
            pool.imap(
                _run_correlation,
                [(np.random.randint(2 ** 32 - 1), it, True) for it in range(n_rounds)],
            )
        ),
        total=n_rounds,
    ):
        tmp_corr = tmp_corr.join(corr_iter.iloc[:, [-1]], rsuffix=f"{i}")

    pool.close()

    pval = [
        (
            100
            - percentileofscore(
                tmp_corr.loc[factor, tmp_corr.columns[-n_rounds:]],
                tmp_corr.loc[factor, "actual_corr"],
            )
        )
        / 100
        for factor in tmp_corr.index
    ]
    tmp_corr["pval"] = pval
    tmp_corr.loc[tmp_corr["actual_corr"] < 0, "pval"] = (
        1 - tmp_corr.loc[tmp_corr["actual_corr"] < 0, "pval"]
    )
    tmp_corr.loc[tmp_corr["pval"] == 0, "pval"] = 1 / (tmp_corr.shape[1] - 2)
    tmp_corr["log_pval"] = -np.log10(tmp_corr["pval"])

    cols = ["std", "abs.actual_corr", "log_pval"]
    rank = pd.DataFrame()
    for col in cols:
        rank[col] = tmp_corr.sort_values(col, ascending=False).index.values
    rank = rankagg(rank)
    rank.columns = ["rank_pval"]
    tmp_corr = tmp_corr.join(rank)

    correlation = correlation.join(tmp_corr[["pval", "log_pval", "rank_pval"]])

    adata.uns["scepia"]["correlation"] = correlation


def plot_volcano_corr(
    adata: AnnData,
    max_pval: Optional[float] = 0.1,
    n_anno: Optional[int] = 40,
    size_anno: Optional[float] = 6,
    palette: Optional[str] = None,
    alpha: Optional[float] = 0.6,
    linewidth: Optional[float] = 0,
    sizes: Optional[Tuple[int, int]] = (1, 20),
    **kwargs,
) -> Axes:
    """Volcano plot of significance of motif-TF correlations.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    """
    if "scepia" not in adata.uns:
        raise ValueError("Motif annotation not found. Did you run `infer_motifs()`?")
    if "correlation" not in adata.uns["scepia"]:
        raise ValueError(
            "Motif-TF correlation data not found. Did you run `determine_significance()`?"
        )

    if palette is None:
        n_colors = len(
            sns.utils.categorical_order(adata.uns["scepia"]["correlation"]["std"])
        )
        cmap = LinearSegmentedColormap.from_list(
            name="grey_black", colors=["grey", "black"]
        )
        palette = sns.color_palette([cmap(i) for i in np.arange(0, 1, 1 / n_colors)])

    sns.set_style("ticks")
    g = sns.scatterplot(
        data=adata.uns["scepia"]["correlation"],
        y="log_pval",
        x="actual_corr",
        size="std",
        hue="std",
        palette=palette,
        sizes=sizes,
        linewidth=linewidth,
        alpha=alpha,
        **kwargs,
    )
    g.legend_.remove()
    g.axhline(y=-np.log10(max_pval), color="grey", zorder=0, ls="dashed")

    c = adata.uns["scepia"]["correlation"]
    factors = c[c["pval"] <= max_pval].sort_values("rank_pval").index[:n_anno]
    x = c.loc[factors, "actual_corr"]
    y = c.loc[factors, "log_pval"]

    texts = []
    for s, xt, yt in zip(factors, x, y):
        texts.append(plt.text(xt, yt, s, {"size": size_anno}))

    x_max = adata.uns["scepia"]["correlation"]["abs.actual_corr"].max() * 1.1
    plt.xlim(-x_max, x_max)
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="black"),
        # expand_points=(1, 1), expand_text=(1, 1),
    )
    plt.xlabel("Correlation (motif vs. factor expression)")
    plt.ylabel("Significance (-log10 p-value)")
    return g
