import os
import sys
import shutil
from tempfile import NamedTemporaryFile
import tarfile

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import pearsonr
from yaml import load
from sklearn.linear_model import MultiTaskLassoCV,BayesianRidge,LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from collections import Counter
from gimmemotifs.motif import read_motifs
from gimmemotifs.moap import moap
from gimmemotifs.maelstrom import run_maelstrom
from gimmemotifs.utils import pwmfile_location
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from appdirs import user_cache_dir
import urllib.request

CACHE_DIR = os.path.join(user_cache_dir("area27"))

def motif_mapping(pfm=None, genes=None, indirect=True):
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
    fname, anno_fname=None, anno_from=None, anno_to=None, scale=False
):
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
    adata,
    gene_df,
    n_neighbors=20,
    center_expression=True,
    model="BayesianRidge",
    use_neighbors=True,
    use_raw=False,
    subsample=True,
):
    """Annotate single cell data.
    """
    # Compute neighborhood graph

    # sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)
    # neighbors = pd.DataFrame(adata.uns['neighbors']['connectivities'].todense())

    # Only use genes that overlap
    common_genes = adata.var_names[adata.var_names.isin(gene_df.index)].unique()
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
    l = getattr(sys.modules[__name__], model)()
    kf = KFold(n_splits=5)

    result = []
    df_coef = pd.DataFrame(index=gene_df.columns)
    for i in tqdm(idxs):
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
                l.fit(X.iloc[idx], y[idx])
                coef = pd.DataFrame(l.coef_, index=gene_df.columns)
                ct = coef.sort_values(0).tail(1).index[0]
                cts.append(ct)
            # df_coef[i] = 0
            top_ct = Counter(cts).most_common()[0][0]
            df_coef[i] = pd.DataFrame.from_dict(Counter(cts), orient="index")
            df_coef[i] = df_coef[i].fillna(0)
        else:
            l.fit(X, y)
            if model == "LogisticRegression":
                coef = pd.DataFrame(l.coef_[0], index=gene_df.columns)
            else:
                coef = pd.DataFrame(l.coef_, index=gene_df.columns)
            df_coef[i] = coef[0]
            top_ct = coef.sort_values(0).tail(1).index[0]
        # print("{}\t{}".format(top_ct, adata.obs['cell_type'].iloc[i]), coef.sort_values(0).tail(5).index)
        result.append([top_ct])

    df_coef = df_coef[sorted(df_coef.columns)]
    return (
        pd.DataFrame(result, columns=["cell_annotation"], index=adata.obs_names[idxs]),
        df_coef,
    )


def relevant_cell_types(adata, gene_df, n_top_genes, cv=5):
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
    print("selecting reference cell types")
    common_genes = list(gene_df.index[gene_df.index.isin(adata.var_names)])
    expression = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names).T
    expression = expression.loc[common_genes]
    expression.columns = adata.obs["louvain"]
    expression = expression.groupby(expression.columns, axis=1).mean()

    var_genes = (
        adata.var.loc[common_genes, "dispersions_norm"].sort_values().tail(n_top_genes).index
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
    print("{} out of {} selected".format(len(cell_types), gene_df.shape[1]))
    print("top 5:")
    for cell_type in cell_types[:5]:
        print(f" * {cell_type}")
    return list(cell_types)


def validate_adata(adata):
    try:
        adata.raw
    except:
        raise ValueError("Please save the raw expression data in the .raw property.")

    if "neighbors" not in adata.uns or "louvain" not in adata.obs:
        raise ValueError("Please run louvain clustering first.")


def load_reference_data(config, data_dir):
    print("loading reference data")
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
    gene_df = read_enhancer_data(fname_genes, anno_fname=anno_fname, anno_to=anno_to)
    return enhancer_df, gene_df


def change_region_size(series, size=200):
    if not isinstance(series, pd.Series):
        try:
            series = series.to_series()
        except:
            series = pd.Series(series)

    loc = series.str.split("[:-]", expand=True)
    loc["start"] = (loc[1].astype(int) + loc[2].astype(int)) // 2 - (size // 2)
    loc["end"] = (loc["start"] + size).astype(str)
    loc["start"] = loc["start"].astype("str")
    return loc[0] + ":" + loc["start"] + "-" + loc["end"]


def infer_motifs(adata, data_dir, pfm=None, min_annotated=50, num_enhancers=10000, maelstrom=False):
    """Infer motif ativity for single cell RNA-seq data.

    The adata object is modified with the following fields.

    **X_cell_types** : `adata.obsm` field
        Cell type coefficients.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    data_dir : `str`
        Directory with reference data.
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

    with open(os.path.join(data_dir, "info.yaml")) as f:
        config = load(f)

    print(config)
    adata.uns["motif"] = {}
    link_file = os.path.join(data_dir, config.get("link_file"))

    gene_map_file = config.get("gene_mapping")
    if gene_map_file is not None:
        gene_map_file = os.path.join(data_dir, gene_map_file)

    enhancer_df, gene_df = load_reference_data(config, data_dir)

    # Determine relevant reference cell types.
    # All other cell types will not be used for motif activity and
    # cell type annotation.
    cell_types = relevant_cell_types(adata, gene_df)
    adata.uns["motif"]["cell_types"] = cell_types

    print("linking variable genes to enhancers")
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
    fname = NamedTemporaryFile().name
    enhancer_df.to_csv(fname, sep="\t")
    print("inferring motif activity")

    pfm = pwmfile_location(pfm)
    adata.uns["motif"]["pfm"] = pfm
    
    
    if maelstrom:
        run_maelstrom(
            fname,
            "hg38",
            "tmp.lala",
            methods=["bayesianridge", "lightningregressor", "xgboost"]
        )

        motif_act = pd.read_csv(os.path.join("tmp.lala", "final.out.csv"), sep="\t", comment="#")
    else:
        motif_act = moap(
            fname,
            scoring="score",
            genome="hg38",
            method="bayesianridge",
            pwmfile=pfm,
            ncpus=12,
        )

    adata.uns["motif"]["motif_activity"] = motif_act[adata.uns["motif"]["cell_types"]]
    print("annotating cells")
    annotation_result, df_coef = annotate_with_k27(
        adata,
        gene_df[cell_types],
        use_neighbors=True,
        model="BayesianRidge",
        subsample=False,
        use_raw=False,
    )
    adata.obsm["X_cell_types"] = df_coef.T[adata.uns["motif"]["cell_types"]].values

    adata.obs = adata.obs.drop(
        columns=annotation_result.columns.intersection(adata.obs.columns)
    )
    adata.obs = adata.obs.join(annotation_result)

    count = adata.obs.groupby("cell_annotation").count().iloc[:, 0]
    valid = count[count > min_annotated].index
    adata.obs["cell_annotation"] = adata.obs["cell_annotation"].astype(str)
    adata.obs.loc[
        ~adata.obs["cell_annotation"].isin(valid), "cell_annotation"
    ] = "other"

    print("calculating cell-specific motif activity")
    cell_motif_activity = (
        adata.uns["motif"]["motif_activity"] @ adata.obsm["X_cell_types"].T
    ).T
    cell_motif_activity.index = adata.obs_names
    adata.obs = adata.obs.drop(
        columns=cell_motif_activity.columns.intersection(adata.obs.columns)
    )
    adata.obs = adata.obs.join(cell_motif_activity)

    correlate_tf_motifs(adata)


def correlate_tf_motifs(adata):
    """Correlate inferred motif activity with TF expression.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    """
    print("correlating TFs with motifs")
    m2f = motif_mapping()
    if issparse(adata.raw.X):
        expression = pd.DataFrame(
            adata.raw.X.todense(), index=adata.obs_names, columns=adata.raw.var_names
        ).T
    else:
        expression = pd.DataFrame(
            adata.raw.X, index=adata.obs_names, columns=adata.raw.var_names
        ).T

    cor = []
    for motif in tqdm(adata.uns["motif"]["motif_activity"].index):
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
    cor["padj"] = multipletests(cor["pval"], method="fdr_bh")[1]
    cor["abscorr"] = np.abs(cor["corr"])
    
    # Determine putative roles of TF based on sign of correlation coefficient.
    # This will not always be correct, as many factors have a high correlation
    # with more than one motif, both positive and negative. In that case it's
    # hard to assign a role.
    cor["putative_role"] = "activator"
    cor.loc[(cor["corr"] + cor["abscorr"]) < 1e-6, "putative_role"] = "repressor"
    adata.uns["motif"]["factor2motif"] = cor


def reassign_cell_types(adata, min_annotated=50):
    adata.obs["cell_annotation"] = (
        pd.Series(adata.uns["motif"]["cell_types"])
        .iloc[adata.obsm["X_cell_types"].argmax(1)]
        .values
    )
    count = adata.obs.groupby("cell_annotation").count().iloc[:, 0]
    valid = count[count > min_annotated].index
    adata.obs["cell_annotation"] = adata.obs["cell_annotation"].astype(str)
    adata.obs.loc[
        ~adata.obs["cell_annotation"].isin(valid), "cell_annotation"
    ] = "other"

def locate_data(dataset, version=None):
    for datadir in [os.path.expanduser(dataset), os.path.join(CACHE_DIR, dataset)]:
        if os.path.isdir(datadir):
            if os.path.exists(os.path.join(datadir, "info.yaml")):
                return datadir
            else:
                raise ValueError(f"info.yaml not found in directory {datadir}")

    # Data file can not be found
    df = pd.read_table("data/data_directory.txt", sep="\t")
    df = df[df["name"] == dataset]
    if df.shape[0] > 0:
        if version is None:
            url = df.sort_values("version").tail(1)["url"].values[0]
        else:
            try:
                url = df.loc[df["version"] == version]["url"].values[0]
            except:
                raise ValueError(f"Dataset {dataset} with version {version} not found.")
        datadir = os.path.join(CACHE_DIR, dataset)
        os.mkdir(datadir)
        datafile = os.path.join(datadir, os.path.split(url)[-1])
        sys.stdout.write(f"Downloading {dataset} data files to {datadir}...\n")
        with urllib.request.urlopen(url) as response, open(datafile, 'wb') as outfile:
            shutil.copyfileobj(response, outfile)
        
        sys.stdout.write("Extracting files...\n")
        tf = tarfile.open(datafile)
        tf.extractall(datadir)
        os.unlink(datafile)

        return datadir
    else:
        raise ValueError(f"Dataset {dataset} not found.")

         
