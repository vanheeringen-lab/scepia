import os
from typing import Optional
import yaml
from pathlib import Path

from loguru import logger
import pandas as pd

from scepia.util import locate_data


class ScepiaDataset:
    def __init__(self, name):

        self.data_dir = Path(locate_data(name))

        with open(os.path.join(self.data_dir, "info.yaml")) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.source = None
        source = self.config.get("source", None)
        if source:
            self.source = ScepiaDataset(source)

    def load_reference_data(self, reftype: Optional[str] = "gene") -> pd.DataFrame:
        logger.info("Loading reference data.")
        fname_enhancers = os.path.join(self.data_dir, self.config["enhancers"])
        fname_genes = os.path.join(self.data_dir, self.config["genes"])
        anno_fname = self.config.get("anno_file")
        if anno_fname is not None:
            anno_fname = os.path.join(self.data_dir, anno_fname)
        anno_to = self.config.get("anno_to")
        anno_from = self.config.get("anno_from")

        if reftype == "enhancer":
            # H3K27ac signal in enhancers
            df = self._read_data_file(
                fname_enhancers, anno_fname=anno_fname, anno_to=anno_to, anno_from=anno_from
            )
        elif reftype == "gene":
            # H3K27ac signal summarized per gene
            df = self._read_data_file(
                fname_genes, anno_fname=anno_fname, anno_to=anno_to, anno_from=anno_from
            )
        else:
            raise ValueError("unknown reference data type")

        if self.source:
            df = df.join(self.source.load_reference_data(reftype=reftype))
        
        return df

    def _read_data_file(
        self,
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

    @property
    def genome(self):
        return self.config.get("genome", "hg38")

    @property
    def meanstd_file(self):
        return self.data_dir / self.config["meanstd_file"]

    @property
    def gene_mapping(self):
        return self.data_dir / self.config.get("gene_mapping")
    
    @property
    def gene_file(self):
        return self.data_dir / self.config.get("gene_file")
    
    @property
    def gene_file(self):
        return self.data_dir / self.config.get("target_file")

    @property
    def version(self):
        return self.config.get("version", "0.0.0")

    @property
    def schema_version(self):
        return self.config.get("schema_version", "0.0.0")
