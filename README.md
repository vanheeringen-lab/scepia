*Please note:* at the moment this package is being actively developed and might not always be stable.

# SCEPIA - Single Cell Epigenome-based Inference of Activity

SCEPIA predicts transcription factor motif activity from single cell RNA-seq data. It uses computationally inferred epigenomes of single cells to identify transcription factors that determine cellular states. The regulatory inference is based on a two-step process:

1) Single cells are matched to a combination of (bulk) reference H3K27ac profiles.
2) Using the H3K27ac signal in enhancers associated with hypervariable genes the TF motif activity is inferred.

The current reference is based on H3K27ac profiles from ENCODE.

## Requirements

* Python >= 3.6
* Scanpy
* GimmeMotifs (development branch)

## Installation

You will need [conda](https://docs.continuum.io/anaconda/) using the [bioconda](https://bioconda.github.io/) channel.

Make sure you have conda installed. If you have not used bioconda before, first set up the necessary channels (in this order!). You only have to do this once.

```
$ conda config --add channels defaults
$ conda config --add channels bioconda
$ conda config --add channels conda-forge
```

Now you can create an environment for scepia:

``` 
conda create -n scepia python=3 adjusttext biofluff gimmemotifs scanpy louvain loguru pyarrow ipywidgets nb_conda
conda activate scepia
```

Install the latest version of GimmeMotifs (which is not yet on biconda):

```
pip install gimmemotifs=0.14.0
```

Install the latest release of scepia:

```
pip install git+https://github.com/vanheeringen-lab/scepia.git@0.3.0
```

## Usage

Remember to activate the environment before using it
```
conda activate scepia
```

### Tutorial

A tutorial on how to use `scepia` can be found [here](tutorials/scepia_tutorial.ipynb).

### Single cell-based motif inference

The [scanpy](https://github.com/theislab/scanpy) package is essential to use scepia. Single cell data should be loaded in an [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) object.
Make sure of the following:

* Gene names are used in `adata.var_names`, not Ensembl identifiers or any other gene identifiers.
* `adata.raw` stores the raw, log-transformed single cell expression data.
* The main `adata` object is filtered to contain only hypervariable genes.
* Louvain clustering has been run.

Once these preprocessing steps are met, `infer_motifs()` can be run to infer the TF motif activity. The first time the reference data will be downloaded, so this will take somewhat longer.

```
from scepia.sc import infer_motifs, determine_significance

# load and preprocess single-cell data using scanpy

adata = infer_motifs(adata, dataset="ENCODE")
determine_significance(adata)
```

The resulting `AnnData` object can be saved with the `.write()` method to a `h5ad` file. However, due to some difficulties with storing the motif annotation in the correct format, the file cannot be loaded with the `scanpy` load() method. Instead, use the `read()` method from the scepia package:

```
from scepia.sc import read
adata = read("my_saved_data.h5ad")
```

The resulting object can now be treated as a normal `AnnData` object.


### Determine enhancer-based regulatory potential

The approach to determine the enhancer-based regulatory potential (ERP) score per gene is based on the approach developed by [Wang et al., 2016](https://dx.doi.org/10.1101%2Fgr.201574.115). There is one difference, in this approach the score is calculates based only on H3K27ac signal in enhancers. We use log-transformed, z-score normalized H3K27ac read counts in 2kb windows centered at enhancer locations. The ERP score is used to match single cell RNA-seq data to the reference H3K27ac profiles.

To use, an H3K27ac BAM file is needed (mapped to hg38). The `-N` argument
specifies the number of threads to use.

```
scepia <bamfile> <outfile> -N 12
```

