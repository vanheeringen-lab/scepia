*Please note:* at the moment this package is being actively developed and might not always be stable.

# SCEPIA - Single Cell Epigenome-based Inference of Activity

SCEPIA predicts transcription factor motif activity from single cell RNA-seq data. It uses computationally inferred epigenomes of single cells to identify transcription factors that determine cellular states. The regulatory inference is based on a two-step process:

1) Single cells are matched to a combination of (bulk) reference H3K27ac ChIP-seq or ATAC-seq profiles.
2) Using the H3K27ac ChIP-seq or ATAC-seq signal in enhancers associated with hypervariable genes the TF motif activity is inferred.

Currently five different references are available, three for human and two for mouse. Different
data sets may give different results, based on a) the type of data (H3K27ac
ChIP-seq or ATAC-seq) and b) the different cell types being represented. While
SCEPIA does not require exact matching cell types to give good results, it does
work best when relatively similar cell types are in the reference. 

The following references can be used:

* `ENCODE.H3K27ac.human` - All H3K27ac experiments from ENCODE. Includes cell
  lines, tissues
* `BLUEPRINT.H3K27ac.human` - All H3K27ac cell types from BLUEPRINT (mostly
  hematopoietic cell types)
* `Domcke.ATAC.fetal.human` - Fetal single cell-based ATAC-seq clusters from
  15 different organs (Domcke et al 2020)[http://dx.doi.org/10.1126/science.aba7612].
* `Cusanovich.ATAC.mouse` - ATAC-seq data of single cell-based clusters from 13
  adult mouse tissues (Cusanovich et al,
2018)[http://dx.doi.org/doi:10.1016/j.cell.2018.06.052].
* `ENCODE.H3K27ac.mouse` - All H3K27ac experiments from mouse ENCODE.

So sorry, but only human and mouse are supported for now. However, if you have data from other species you can try it if gene names tend to match. Make sure you usegene names as identifiers, and `scepia` will run fine. In our (very limited) experience this *can* yield good results, but there are a lot of assumptions on conservation of regulatory interactions. If you have a large collection of ATAC-seq or ChIP-seq reference experiments available you can also create your own reference with `ScepiaDataset.create()`. This is not well-documented at the moment, let us know if you need help to do so.

## Requirements and installation

You will need [conda](https://docs.continuum.io/anaconda/) using the [bioconda](https://bioconda.github.io/) channel.

Make sure you have conda installed. If you have not used bioconda before, first set up the necessary channels (in this order!). You only have to do this once.

```
$ conda config --add channels defaults
$ conda config --add channels bioconda
$ conda config --add channels conda-forge
```

Now you can create an environment for scepia:

``` 
conda create -n scepia adjusttext biofluff gimmemotifs>=0.15.1 scanpy leidenalg louvain loguru geosketch
# Note: if you want to use scepia in a Jupyter notebook, you also have to install the following packages: `ipywidgets nb_conda`.
conda activate scepia
```

Install the latest release of scepia:

```
pip install git+https://github.com/vanheeringen-lab/scepia.git@0.3.5
```

## Usage

### Command line

Remember to activate the environment before using it

```
conda activate scepia
```

The command line script `scepia infer_motifs` works on any file that is supported by [`scanpy.read()`](https://scanpy.readthedocs.io/en/stable/api/scanpy.read.html). We recommend to process your data, including QC, filtering, normalization and clustering, using scanpy. If you save the results to an `.h5ad` file, `scepia` can continue from your analysis to infer motif activity. However, the command line tool also works on formats such as CSV files or tab-separated files. In that case, `scepia` will run some basic pre-processing steps. To run `scepia`:

```
scepia infer_motifs <input_file> <output_dir>
```

### Jupyter notebook tutorial

A tutorial on how to use `scepia` interactively in Jupyter can be found [here](tutorials/scepia_tutorial.ipynb).

Single cell data should be loaded in an [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) object.
Make sure of the following:

* Gene names are used in `adata.var_names`, not Ensembl identifiers or any other gene identifiers.
* `adata.raw` stores the raw, log-transformed single cell expression data.
* The main `adata` object is filtered to contain only hypervariable genes.
* Louvain clustering has been run.

Once these preprocessing steps are met, `infer_motifs()` can be run to infer the TF motif activity. The first time the reference data will be downloaded, so this will take somewhat longer.

```
from scepia.sc import infer_motifs

# load and preprocess single-cell data using scanpy

adata = infer_motifs(adata, dataset="ENCODE")
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
scepia area27 <bamfile> <outfile> -N 12
```

