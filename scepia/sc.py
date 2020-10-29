# Copyright (c) 2019 Simon van Heeringen <simon.vanheeringen@gmail.com>
#
# This module is free software. You can redistribute it and/or modify it under
# the terms of the MIT License, see the file LICENSE included with this
# distribution.
from collections import Counter
import os
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

# Typing
from typing import List, Optional, Tuple

from anndata import AnnData
from appdirs import user_cache_dir

from loguru import logger
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LassoCV
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
from scipy.stats import percentileofscore, combine_pvalues
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
from yaml import load
import geosketch

from gimmemotifs.moap import moap
from gimmemotifs.maelstrom import run_maelstrom
from gimmemotifs.motif import read_motifs
from gimmemotifs.utils import pfmfile_location
from scepia import __version__
from scepia.plot import plot
from scepia.util import fast_corr
from scepia.data import ScepiaDataset

CACHE_DIR = os.path.join(user_cache_dir("scepia"))


class MotifAnnData(AnnData):
    """Extended AnnData class.

    Add the functionality to correctly save and load an AnnData object with
    motif annotation results.
    """

    df_keys = ["motif_activity", "correlation"]

    def __init__(self, adata):
        super().__init__(adata)

    def _remove_additional_data(self) -> None:
        # Motif columns need to be removed, as the hdf5 backend cannot
        # store the data otherwise (header too long).
        self.obs = self.obs.drop(
            columns=self.uns["scepia"]
            .get("motif_activity", pd.DataFrame())
            .columns.intersection(self.obs.columns)
        )
        # DataFrames are not supported in the h5ad format. By converting them
        # dictionaries the can be restored to DataFrame format after loading.
        for k in self.df_keys:
            if k not in self.uns["scepia"]:
                continue
            logger.info(f"updating {k}")
            self.uns["scepia"][f"{k}_columns"] = self.uns["scepia"][k].columns.tolist()
            self.uns["scepia"][f"{k}_index"] = self.uns["scepia"][k].index.tolist()
            self.uns["scepia"][k] = self.uns["scepia"][k].to_numpy()

        adata.uns['scepia']['cell_types'] = adata.uns['scepia']['cell_types'].tolist()

    def _restore_additional_data(self) -> None:
        # In this case it works for an AnnData object that contains no
        # additional motif information
        if "scepia" not in self.uns:
            return

        # Restore all DataFrames in uns
        logger.info("Restoring DataFrames")
        for k in self.df_keys:
            if k not in self.uns["scepia"]:
                continue
            self.uns["scepia"][k] = pd.DataFrame(
                self.uns["scepia"][k],
                index=self.uns["scepia"][f"{k}_index"],
                columns=self.uns["scepia"][f"{k}_columns"],
            )

        for k in self.df_keys + ["cell_types"]:
            if k not in self.uns["scepia"]:
                logger.warning("scepia information is not complete")
                return

        for k in self.df_keys:
            for col in self.uns["scepia"][k].columns:
                try:
                    self.uns["scepia"][k][col] = self.uns["scepia"][k][col].astype(
                        float
                    )
                except Exception:
                    pass

        # make sure index has the correct name
        self.uns["scepia"]["correlation"].index.rename("factor", inplace=True)

        # Make sure the cell types are in the correct order
        logger.info("Sorting cell types")
        self.uns["scepia"]["motif_activity"] = self.uns["scepia"]["motif_activity"][
            self.uns["scepia"]["cell_types"]
        ]

        if "X_cell_types" not in self.obsm:
            logger.warning("scepia information is not complete")

        #  The cell type-specific motif activity needs to be recreated.
        logger.info("Recreate motif activity")
        cell_motif_activity = pd.DataFrame(
            self.uns["scepia"]["motif_activity"] @ self.obsm["X_cell_types"].T
        ).T
        logger.info("Drop columns")
        cell_motif_activity.index = self.obs_names
        self.obs = self.obs.drop(
            columns=cell_motif_activity.columns.intersection(self.obs.columns)
        )
        logger.info("Add motif activity to obs")
        self.obs = self.obs.join(cell_motif_activity)

    def write(self, *args, **kwargs) -> None:
        """Write a MotifAnnData object.

        All DataFrames in uns are converted to dictionaries and motif columns
        are removed from obs.
        """
        logger.info("writing scepia-compatible file")
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


def annotate_with_k27(
    adata: AnnData,
    gene_df: pd.DataFrame,
    cluster: Optional[str] = "louvain",
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
        expression = adata.X[:, adata.var_names.isin(gene_df.index)]
        expression = (
            np.squeeze(np.asarray(expression.todense())) if issparse(expression) else expression
        )
        expression = pd.DataFrame(
            expression,
            index=adata.obs_names,
            columns=common_genes,
        ).T

    if center_expression:
        expression = expression.sub(expression.mean(1), 0)

    # Get sampled idxs
    N = 100000
    unique_cell_types = adata.obs[cluster].unique()
    counts = adata.obs.groupby(cluster).count().iloc[:, 0].to_dict()
    ids = np.arange(adata.shape[0])
    idxs = []
    for cell_type in unique_cell_types:
        if counts[cell_type] <= N:
            idx = ids[adata.obs[cluster] == cell_type]
        else:
            idx = np.random.choice(
                ids[adata.obs[cluster] == cell_type], N, replace=False
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
                    pd.DataFrame((adata.obsp["connectivities"][i] != 0).todense())
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
    cluster: Optional[str] = "louvain",
    n_top_genes: Optional[int] = 1000,
    max_cell_types: Optional[int] = 50,
    cv: Optional[int] = 5,
) -> List[str]:
    """Select relevant cell types for annotation and motif inference.

    Based on Lasso regression a subset of features (cell type
    profile) will be selected. Expression is averaged over clusters.
    Requires louvain or leiden clustering to be run on the `adata` object.

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
    max_cell_types : `int`, optional (default: 50)
        Maximum number of cell types to select.
    cv : `int`, optional (default: 5)
        Folds for cross-validation

    Returns
    -------
    `list`
        Cell types ordered by the mean absolute coefficient over clusters in
        descending order.
    """
    logger.info("selecting reference cell types")
    common_genes = list(gene_df.index[gene_df.index.isin(adata.var_names)])

    expression = (
        np.squeeze(np.asarray(adata.X.todense())) if issparse(adata.X) else adata.X
    )

    expression = pd.DataFrame(
        expression, index=adata.obs_names, columns=adata.var_names
    ).T
    expression = expression.loc[common_genes]
    expression.columns = adata.obs[cluster]
    expression = expression.groupby(expression.columns, axis=1).mean()

    var_genes = (
        adata.var.loc[common_genes, "dispersions_norm"]
        .sort_values()
        .tail(n_top_genes)
        .index
    )
    expression = expression.loc[var_genes]
    X = gene_df.loc[var_genes]
    g = LassoCV(cv=cv, selection="random")
    cell_types = pd.DataFrame(index=X.columns)
    
    for col in expression.columns:
        g.fit(X, expression[col])
        coefs = pd.DataFrame(g.coef_, index=X.columns)
        cell_types[col] = coefs
    
    cell_types = cell_types.abs().sum(1).sort_values().tail(max_cell_types)
    cell_types = cell_types[cell_types > 0].index
    top = cell_types[-5:]
    
    logger.info("{} out of {} selected".format(len(cell_types), gene_df.shape[1]))
    logger.info(f"Top {len(top)}:")
    for cell_type in top:
        logger.info(f" * {cell_type}")
    return cell_types


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

    if "connectivities" not in adata.obsp or (
        "louvain" not in adata.obs and "leiden" not in adata.obs
    ):
        raise ValueError("Please run louvain or leiden clustering first.")


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


def annotate_cells(
    adata: AnnData,
    dataset: str,
    cluster: Optional[str] = "louvain",
    n_top_genes: Optional[int] = 1000,
    max_cell_types: Optional[int] = 50,
    min_annotated: Optional[int] = 50,
    select: Optional[bool] = True,
) -> None:
    """
    Assign cells with cell type based on H3K27ac reference profiles.
    """
    # Determine relevant reference cell types.
    # All other cell types will not be used for motif activity and
    # cell type annotation.
    data = ScepiaDataset(dataset)
    gene_df = data.load_reference_data(reftype="gene")

    if select:
        cell_types = relevant_cell_types(
            adata, gene_df, cluster=cluster, n_top_genes=n_top_genes, max_cell_types=max_cell_types,
        )
    else:
        logger.info("Selecting all reference cell types.")
        cell_types = gene_df.columns

    if "scepia" not in adata.uns:
        adata.uns["scepia"] = {"version": __version__}

    adata.uns["scepia"]["cell_types"] = cell_types

    logger.info("Annotating cells.")
    annotation_result, df_coef = annotate_with_k27(
        adata,
        gene_df[cell_types],
        cluster=cluster,
        use_neighbors=True,
        model="BayesianRidge",
        subsample=False,
        use_raw=False,
    )
    adata.obsm["X_cell_types"] = df_coef.T[adata.uns["scepia"]["cell_types"]].values

    # Annotate by highest mean coefficient
    coefs = pd.DataFrame(
        adata.obsm["X_cell_types"], index=adata.obs_names, columns=cell_types
    )
    coefs["cluster"] = adata.obs[cluster]
    cluster_anno = (
        coefs.groupby("cluster").mean().idxmax(axis=1).to_frame("cluster_annotation")
    )

    if "cluster_annotation" in adata.obs:
        adata.obs = adata.obs.drop(columns=["cluster_annotation"])

    adata.obs = adata.obs.join(cluster_anno, on=cluster)

    # Second round of annotation, including "other"
    assign_cell_types(adata, min_annotated=min_annotated)


def infer_motifs(
    adata: AnnData,
    dataset: str,
    cluster: Optional[str] = "louvain",
    n_top_genes: Optional[int] = 1000,
    max_cell_types: Optional[int] = 50,
    pfm: Optional[str] = None,
    min_annotated: Optional[int] = 50,
    num_enhancers: Optional[int] = 10000,
    maelstrom: Optional[bool] = False,
    indirect: Optional[bool] = True,
    n_sketch: Optional[int] = 2500,
    n_permutations: Optional[int] = 100000,
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
    cluster : `str`, optional (default: "louvain")
        Name of the clustering, can be either louvain or leiden.
    n_top_genes : `int`, optional (default: 1000)
        Number of variable genes that is used. If `n_top_genes` is greater than the
        number of hypervariable genes in `adata` then all variable genes are
        used.
    max_cell_types : `int`, optional (default: 50)
        Maximum number of cell types to select.
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

    data = ScepiaDataset(dataset)

    logger.debug(config)
    if "scepia" not in adata.uns:
        adata.uns["scepia"] = {"version": __version__}

    # Annotate each cell with H3K27ac reference
    if "cell_annotation" not in adata.obs or "cluster_annotation" not in adata.obs:
        annotate_cells(
            adata,
            dataset=dataset,
            cluster=cluster,
            n_top_genes=n_top_genes,
            min_annotated=min_annotated,
            max_cell_types=max_cell_types,
        )

    logger.info("Linking variable genes to differential enhancers.")
    gene_map_file = config.get("gene_mapping")
    if gene_map_file is not None:
        gene_map_file = os.path.join(data_dir, gene_map_file)

    link_file = os.path.join(data_dir, config.get("link_file"))
    link = pd.read_feather(link_file)
    if use_name:
        ens2name = pd.read_csv(
            gene_map_file, sep="\t", index_col=0, names=["identifier", "name"]
        )
        link = link.join(ens2name, on="gene").dropna()
        link = link.set_index("name")

    enh_genes = adata.var_names[adata.var_names.isin(link.index)]
    var_enhancers = change_region_size(link.loc[enh_genes, "loc"]).unique()

    enhancer_df = data.load_reference_data(reftype="enhancer")
    enhancer_df.index = change_region_size(enhancer_df.index)
    enhancer_df = enhancer_df.loc[var_enhancers, adata.uns["scepia"]["cell_types"]]
    enhancer_df = enhancer_df.groupby(enhancer_df.columns, axis=1).mean()
    enhancer_df.loc[:, :] = scale(enhancer_df)
    # Select top most variable enhancers
    enhancer_df = enhancer_df.loc[
        enhancer_df.var(1).sort_values().tail(num_enhancers).index
    ]
    # Center by mean of the most import cell types
    # Here we chose the majority cell type per cluster
    cluster_cell_types = adata.obs["cluster_annotation"].unique()
    mean_value = enhancer_df[cluster_cell_types].mean(1)
    enhancer_df = enhancer_df.sub(mean_value, axis=0)
    fname = NamedTemporaryFile(delete=False).name
    enhancer_df.to_csv(fname, sep="\t")
    logger.info("inferring motif activity")

    pfm = pfmfile_location(pfm)
    if maelstrom:
        with TemporaryDirectory() as tmpdir:
            run_maelstrom(
                fname, data.genome, tmpdir, center=False, filter_redundant=True,
            )

            motif_act = pd.read_csv(
                os.path.join(tmpdir, "final.out.txt"),
                sep="\t",
                comment="#",
                index_col=0,
            )
            motif_act.columns = motif_act.columns.str.replace(r"z-score\s+", "")
            pfm = pfmfile_location(os.path.join(tmpdir, "nonredundant.motifs.pfm"))
    else:
        motif_act = moap(
            fname,
            scoring="score",
            genome="hg38",
            method="bayesianridge",
            pfmfile=pfm,
            ncpus=12,
        )
    adata.uns["scepia"]["pfm"] = pfm

    adata.uns["scepia"]["motif_activity"] = motif_act[adata.uns["scepia"]["cell_types"]]

    logger.info("calculating cell-specific motif activity")
    cell_motif_activity = (
        adata.uns["scepia"]["motif_activity"] @ adata.obsm["X_cell_types"].T
    ).T
    cell_motif_activity.index = adata.obs_names
    adata.obs = adata.obs.drop(
        columns=cell_motif_activity.columns.intersection(adata.obs.columns)
    )
    adata.obs = adata.obs.join(cell_motif_activity)

    correlate_tf_motifs(
        adata, indirect=indirect, n_sketch=n_sketch, n_permutations=n_permutations
    )

    add_activity(adata)

    logger.info("Done with motif inference.")
    return MotifAnnData(adata)


def correlate_tf_motifs(
    adata: AnnData,
    n_sketch: Optional[int] = 2500,
    n_permutations: Optional[int] = 100000,
    indirect: Optional[bool] = True,
) -> None:
    """Correlate inferred motif activity with TF expression.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    n_sketch : `int`, optional (default: 2500)
        If the number of cells is higher than `n_sketch`, use geometric
        sketching (Hie et al. 2019) to select a subset of `n_sketch`
        cells. This subset will be used to calculate the correlation beteen
        motif activity and transcription factor expression.
    n_permutations : `int`, optional (default: 100000)
        Number of permutations that is used to calculate the p-value. Can be
        decreased for quicker run-time, but should probably not be below 10000.
    indirect : `bool`, optional (default: True)
        Include indirect TF to motif assignments.
    """
    logger.info("correlating motif activity with factors")
    if indirect:
        logger.info("including indirect and/or predicted factors")
    # Get all TFs from motif database
    m2f = motif_mapping(indirect=True)
    batch_size = m2f.shape[0]
    f2m2 = pd.DataFrame(m2f["factors"].str.split(",").tolist(), index=m2f.index).stack()
    f2m2 = f2m2.to_frame().reset_index().iloc[:, [0, 2]]
    f2m2.columns = ["motif", "factor"]
    unique_factors = f2m2["factor"].unique()

    if n_sketch is None or n_sketch > adata.shape[0]:
        logger.info(f"using all cells")
        my_adata = adata
    else:
        logger.info(f"creating sketch of {n_sketch} cells")
        idx = geosketch.gs(adata.obsm["X_pca"], n_sketch)
        my_adata = adata.copy()
        my_adata = my_adata[idx]

    detected = (my_adata.raw.var_names.str.upper().isin(unique_factors)) & (
        (my_adata.raw.X > 0).sum(0) > 3
    )
    detected = np.squeeze(np.asarray(detected))
    unique_factors = my_adata.raw.var_names[detected].str.upper()

    # Get the expression for all TFs
    expression = (
        np.squeeze(np.asarray(my_adata.raw.X.todense()))
        if issparse(my_adata.raw.X)
        else my_adata.raw.X
    )
    expression = expression.T[detected]

    logger.info(
        f"calculating correlation of motif activity with {len(unique_factors)} factors"
    )
    real = fast_corr(
        expression,
        (
            my_adata.obsm["X_cell_types"] @ my_adata.uns["scepia"]["motif_activity"].T
        ).T.values,
    )
    real = pd.DataFrame(
        real,
        index=unique_factors,
        columns=my_adata.uns["scepia"]["motif_activity"].index,
    )

    tmp = (
        real.reset_index()
        .melt(id_vars="index", var_name="motif", value_name="correlation")
        .rename(columns={"index": "factor"})
        .set_index(["motif", "factor"])
    )
    f2m2 = f2m2.set_index(["motif", "factor"]).join(tmp).dropna()
    f2m2["abs_correlation"] = f2m2["correlation"].abs()

    logger.info(f"calculating {n_permutations} permutations")
    permute_result = pd.DataFrame(index=unique_factors)
    shape = my_adata.uns["scepia"]["motif_activity"].shape
    for i in tqdm(range(0, n_permutations, batch_size)):
        random_activities = None
        while random_activities is None or random_activities.shape[0] < batch_size:
            x = my_adata.uns["scepia"]["motif_activity"].values.flatten()
            motif_activity = shuffle(x).reshape(shape[1], shape[0])
            cell_motif_activity = (my_adata.obsm["X_cell_types"] @ motif_activity).T
            if random_activities is None:
                random_activities = cell_motif_activity
            else:
                random_activities = np.vstack((random_activities, cell_motif_activity))

        random_activities = random_activities[:batch_size]
        batch_result = fast_corr(expression, random_activities)
        batch_result = pd.DataFrame(
            batch_result, index=unique_factors, columns=range(i, i + batch_size)
        )
        permute_result = permute_result.join(batch_result)

    logger.info("calculating permutation-based p-values (all)")

    # Calculate p-value of correlation relative to all permuted correlations
    permuted_corrs = permute_result.values.flatten()
    pvals = [
        (100 - percentileofscore(permuted_corrs, corr)) / 100
        for corr in f2m2["correlation"]
    ]
    f2m2["pval"] = pvals
    f2m2.loc[f2m2["correlation"] < 0, "pval"] = (
        1 - f2m2.loc[f2m2["correlation"] < 0, "pval"]
    )
    logger.info("calculating permutation-based p-values (factor-specific)")

    # Calculate p-value of correlation relative to permutated value of this factor
    for motif, factor in tqdm(f2m2.index):
        pval = (
            100 - percentileofscore(permute_result.loc[factor], real.loc[factor, motif])
        ) / 100
        pval = 1 - pval if real.loc[factor, motif] < 0 else pval
        pval = 1 / permute_result.shape[1] if pval == 0 else pval
        f2m2.loc[(motif, factor), "permutation_pval"] = pval
        f2m2.loc[(motif, factor), "combined"] = combine_pvalues(
            f2m2.loc[(motif, factor), ["pval", "permutation_pval"]]
        )[1]

    f2m2["p_adj"] = multipletests(f2m2["combined"], method="fdr_bh")[1]
    f2m2["-log10(p-value)"] = -np.log10(f2m2["p_adj"])

    cluster_cell_types = adata.obs["cluster_annotation"].unique()
    f2m2 = f2m2.join(
        (
            adata.uns["scepia"]["motif_activity"][cluster_cell_types].max(1)
            - adata.uns["scepia"]["motif_activity"][cluster_cell_types].min(1)
        )
        .to_frame("motif_stddev")
        .rename_axis("motif")
    )

    f2m2 = f2m2.reset_index().set_index("factor")
    adata.uns["scepia"]["correlation"] = f2m2


def add_activity(adata: AnnData):
    """Get factor activity"""
    if "scepia" not in adata.uns:
        raise ValueError(
            "Could not find motif information. Did you run infer_motifs() first?"
        )
    if "correlation" not in adata.uns["scepia"]:
        logger.warn("Cannot determine factor activity without factor annotation")
        return

    gm = GaussianMixture(n_components=2, covariance_type="full")
    f2m = adata.uns["scepia"]["correlation"]
    logger.info("Inferring factor activity.")
    for factor in tqdm(f2m.index.unique()):
        motif = f2m.loc[[factor]].sort_values("p_adj").iloc[0].motif
        gm.fit(adata.obs[motif].values.reshape(-1, 1) * 10)
        adata.obs[f"{factor}_activity"] = gm.predict_proba(adata.obs[[motif]] * 10)[
            :, gm.means_.argmax()
        ]


def assign_cell_types(adata: AnnData, min_annotated: Optional[int] = 50) -> None:
    # adata.obs["cell_annotation"] = (
    #    pd.Series(adata.uns["scepia"]["cell_types"])
    #    .iloc[adata.obsm["X_cell_types"].argmax(1)]
    #    .values
    # )
    neighbour_coef = adata.obsp["connectivities"] @ adata.obsm["X_cell_types"]
    neighbour_coef = pd.DataFrame(
        neighbour_coef, index=adata.obs_names, columns=adata.uns["scepia"]["cell_types"]
    )
    adata.obs["cell_annotation"] = neighbour_coef.idxmax(axis=1)
    count = adata.obs.groupby("cell_annotation").count().iloc[:, 0]
    valid = count[count > min_annotated].index
    adata.obs["cell_annotation"] = adata.obs["cell_annotation"].astype(str)
    adata.obs.loc[
        ~adata.obs["cell_annotation"].isin(valid), "cell_annotation"
    ] = "other"


def _simple_preprocess(adata: AnnData,) -> AnnData:

    logger.info("Running a simple preprocessing pipeline based on scanpy docs.")
    logger.info(
        "To control this process, run the analysis in scanpy, save the h5ad file and analyze this file with scepia."
    )

    logger.info("Filtering cells and genes.")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    logger.info("Normalizing and log-transforming.")
    sc.pp.normalize_total(adata, target_sum=1e4)

    sc.pp.log1p(adata)
    adata.raw = adata

    logger.info("Selecting highly variable genes.")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    logger.info("Regressing out total counts.")
    sc.pp.regress_out(adata, ["total_counts"])

    logger.info("Scaling.")
    sc.pp.scale(adata, max_value=10)

    logger.info("Running PCA.")
    sc.tl.pca(adata, svd_solver="arpack")

    logger.info("Running nearest neighbors.")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    logger.info("Running Leiden clustering.")
    sc.tl.leiden(adata)

    logger.info("Running UMAP")
    sc.tl.umap(adata)

    return adata


def full_analysis(
    infile: str,
    outdir: str,
    ftype: Optional[str] = "auto",
    transpose: Optional[bool] = False,
    cluster: Optional[str] = None,
    n_top_genes: Optional[int] = 1000,
    pfmfile: Optional[str] = None,
    min_annotated: Optional[int] = 50,
    num_enhancers: Optional[int] = 10000,
):
    """
    Run full SCEPIA analysis on h5ad infile.
    """
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    infile = os.path.expanduser(infile)
    basename = os.path.basename(infile)
    basename = os.path.splitext(basename)[0]
    outfile = os.path.join(outdir, f"{basename}.scepia.h5ad")

    logfile = os.path.join(outdir, "scepia.log")
    logger.add(logfile, level="DEBUG", mode="w")

    logger.info(f"Reading {infile} using scanpy.")
    if os.path.isdir(infile):
        try:
            logger.debug(f"Trying 10x mtx directory.")
            adata = sc.read_10x_mtx(infile)
            adata.obs_names_make_unique()
        except Exception as e:
            logger.debug(f"Failed: {str(e)}.")
            logger.debug(f"Fallback to normal sc.read().")
            adata = sc.read(infile)
    else:
        adata = sc.read(infile)

    if transpose:
        logger.info("Transposing matrix.")
        adata = adata.T

    logger.info(f"{adata.shape[0]} cells x {adata.shape[1]} genes")
    if not transpose:
        logger.info(
            "(Cells and genes mixed up? Try transposing your data by adding the --transpose argument.)"
        )

    if adata.raw is None or "connectivities" not in adata.obsp:
        logger.info("No processed information found (connectivity graph, clustering).")
        logger.info("Running basic preprocessing analysis.")
        adata = _simple_preprocess(adata)

    if cluster is None:
        if "leiden" in adata.obs:
            cluster = "leiden"
        else:
            cluster = "louvain"

    adata = infer_motifs(
        adata,
        dataset="ENCODE",
        cluster=cluster,
        n_top_genes=n_top_genes,
        pfm=pfmfile,
        min_annotated=min_annotated,
        num_enhancers=num_enhancers,
        indirect=True,
    )
    f2m = os.path.join(outdir, "factor_motif_correlation.txt")
    adata.uns["scepia"]["correlation"].to_csv(f2m, sep="\t")
    adata.write(outfile)

    fname = os.path.join(outdir, "cell_x_motif_activity.txt")
    cellxact = adata.uns["scepia"]["motif_activity"] @ adata.obsm["X_cell_types"].T
    cellxact.columns = adata.obs_names
    cellxact.to_csv(fname, sep="\t")

    fname = os.path.join(outdir, "cell_properties.txt")
    columns = (
        ["cell_annotation", "cluster_annotation"]
        + (adata.uns["scepia"]["motif_activity"].index).tolist()
        + adata.obs.columns[adata.obs.columns.str.contains("_activity$")].tolist()
    )
    adata.obs[columns].to_csv(fname, sep="\t")

    adata.write(outfile)

    fig = plot(adata, n_anno=40)
    fig.savefig(os.path.join(outdir, "volcano.png"), dpi=600)
