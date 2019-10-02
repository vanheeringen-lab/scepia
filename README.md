*Note:* at the moment this package is being actively developed and might not always be stable.

# Area27

Inference of transcription factor motif activity from single cell RNA-seq data. The regulatory inference is based on a two-step process:

1) Single cells are matched to a combination of (bulk) reference H3K27ac profiles.
2) Using the H3K27ac signal in enhancers associated with hypervariable genes the TF motif activity is inferred.

The current reference is based on H3K27ac profiles from ENCODE.

## Requirements

* Python >= 3.6

## Installation

``` 
conda create -n area27 python=3 biofluff gimmemotifs scanpy
conda activate area27
pip install git+https://github.com/vanheeringen-lab/gimmemotifs.git@develop
pip install git+https://github.com/vanheeringen-lab/area27.git@develop
```

## Usage

Remember to activate the environment before using it
```
conda activate area27
```

### Single cell-based motif inference

The [scanpy](https://github.com/theislab/scanpy) package is essential to use area27. Single cell data should be loaded in an [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) object.
Make sure of the following:

* `adata.raw` stores the raw, log-transformed single cell expression data.
* The main `adata` object is filtered to contain only hypervariable genes.
* Louvain clustering has been run.

Once these preprocessing steps are met `infer_motifs()` can be run to infer the TF motif activity.

```
from area27.sc import infer_motifs

# load preprocess single-cell data

infer_motifs(adata, data_dir="/path/to/ENCODE_dir")
```

### Determine enhancer-based regulatory potential

To use, an H3K27ac BAM file is needed (mapped to hg38). The `-N` argument
specifies the number of threads to use.

```
area27 <bamfile> <outfile> -N 12
```

