# Area27

Compare H3K27ac profile with reference profiles to highlight genes with higher
signal than average.


## Requirements

* Python >= 3.6

## Installation

``` 
conda create -n area27 python=3 pandas xdg biofluff genomepy 
conda activate area27
pip install git+https://github.com/vanheeringen-lab/area27.git
```

## Usage

Remember to activate the environment before using it
```
conda activate area27
```

To use, an H3K27ac BAM file is needed (mapped to hg38). The `-N` argument
specifies the number of threads to use.

```
area7 <bamfile> <outfile> -N 12
```

