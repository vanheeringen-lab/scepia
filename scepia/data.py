import os
from typing import Optional,List,Type,TypeVar

from pathlib import Path
from tempfile import NamedTemporaryFile

from genomepy import Genome
from gimmemotifs.preprocessing import coverage_table
from loguru import logger
import numpy as np
import pandas as pd
from pybedtools import BedTool
from shutil import copyfile
from tqdm.auto import tqdm
import qnorm
import yaml

from scepia.util import locate_data
from scepia import create_link_file, generate_signal, link_it_up

__schema_version__ = "0.1.0"
T = TypeVar('T', bound='ScepiaDataset')

def _create_gene_table(
    df: pd.DataFrame,
    meanstd_file: str,
    gene_file: str,
    gene_mapping: str,
    genome: Optional[str] = None,
    link_file: Optional[str] = None,
    threshold: Optional[float] = 1.0,
):
    logger.info("Calculating gene-based values")
    genes = None
    for exp in tqdm(df.columns):
        tmp = link_it_up(
            df[exp].to_frame("signal"),
            meanstd_file=meanstd_file,
            genes_file=gene_file,
            names_file=gene_mapping,
            genome=genome,
            link_file=link_file,
            threshold=threshold,
        )
        tmp.columns = [exp]
        tmp = tmp.sort_values(exp)
        tmp = tmp[~tmp.index.duplicated(keep="last")]

        if genes is None:
            genes = tmp
        else:
            genes = pd.concat((genes, tmp), axis=1)  # genes.join(tmp, how="outer")

    genes = genes.fillna(0)
    return genes


class ScepiaDataset:
    def __init__(self, name):
        self.name = str(name)
        self.data_dir = Path(locate_data(name))

        with open(os.path.join(self.data_dir, "info.yaml")) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.source = None
        source = self.config.get("source", None)
        if source:
            self.source = ScepiaDataset(source)

    @property
    def genome(self):
        return self.config.get("genome", "hg38")

    @property
    def window(self):
        if self.source:
            return self.source.window
        else:
            return int(self.config.get("window", 2000))

    @property
    def meanstd_file(self):
        if self.source:
            return self.source.meanstd_file
        else:
            return self.data_dir / self.config["meanstd_file"]

    @property
    def gene_mapping(self):
        if self.source:
            return self.source.gene_mapping
        else:
            return self.data_dir / self.config.get("gene_mapping")

    @property
    def gene_file(self):
        if self.source:
            return self.source.gene_file
        else:
            return self.data_dir / self.config.get("gene_file")

    @property
    def target_file(self):
        if self.source:
            return self.source.target_file
        else:
            return self.data_dir / self.config.get("target_file")

    @property
    def link_file(self):
        if self.source:
            return self.source.link_file
        else:
            return self.data_dir / self.config.get("link_file")
    
    @property
    def version(self):
        return self.config.get("version", "0.0.0")

    @property
    def schema_version(self):
        return self.config.get("schema_version", "0.0.0")

    def load_reference_data(
        self, reftype: Optional[str] = "gene", scale: Optional[bool] = True
    ) -> pd.DataFrame:
        
        if reftype not in ["enhancer", "gene"]:
            raise ValueError("unknown reference data type")

        logger.info(f"Loading reference data ({reftype}).")
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
                fname_enhancers,
                anno_fname=anno_fname,
                anno_to=anno_to,
                anno_from=anno_from,
            )
        elif reftype == "gene":
            # H3K27ac signal summarized per gene
            df = self._read_data_file(
                fname_genes, anno_fname=anno_fname, anno_to=anno_to, anno_from=anno_from
            )

        if self.source:
            df = df.join(self.source.load_reference_data(reftype=reftype, scale=False))
        
        df = df[df.max(1) > 0]
        df = df.fillna(0)

        if scale:
            df = df.sub(df.mean(1), axis=0)
            df = df.div(df.std(1), axis=0)

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

    @classmethod
    def create(
        cls: Type[T],
        outdir: str,
        data_files: List[str],
        enhancer_file: str,
        annotation_file: str,
        genome: str,
        window: Optional[int] = 2000,
        anno_file: Optional[str] = None,
        anno_from: Optional[str] = None,
        anno_to: Optional[str] = None,
        gene_mapping: Optional[str] = None,
        threshold: Optional[float] = 1.0,
        version: Optional[str] = "0.1.0",
    ) -> T:
        outdir = Path(outdir)
        basename = outdir.name
        meanstd_file = outdir / f"{basename}.{genome}.meanstd.tsv.gz"
        target_file = outdir / f"{basename}.{genome}.target.npz"
        gene_file = outdir / "annotation.tss.merged1kb.bed"
        link_file = outdir / "enhancers2genes.feather"

        g = Genome(genome)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        info = {
            "genes": "genes.txt",
            "enhancers": "enhancers.feather",
            "link_file": os.path.basename(link_file),
            "genome": genome,
            "window": window,
            "meanstd_file": os.path.basename(meanstd_file),
            "target_file": os.path.basename(target_file),
            "gene_file": os.path.basename(gene_file),
            "version": version,
            "schema_version": __schema_version__,
        }

        if anno_file is not None:
            if not os.path.exists(anno_file):
                raise ValueError(f"{anno_file} does not exist")
            if anno_from is None or anno_to is None:
                raise ValueError("Need anno_from and anno_to columns!")
            copyfile(anno_file, outdir / os.path.basename(anno_file))
            info.update(
                {
                    "anno_file": os.path.basename(anno_file),
                    "anno_from": anno_from,
                    "anno_to": anno_to,
                }
            )

        if gene_mapping is not None:
            if not os.path.exists(gene_mapping):
                raise ValueError(f"{gene_mapping} does not exist")
            copyfile(gene_mapping, outdir / os.path.basename(gene_mapping))
            info["gene_mapping"] = os.path.basename(gene_mapping)

        logger.info("processing gene annotation")
        # Convert gene annotation
        b = BedTool(annotation_file)
        chroms = set([f.chrom for f in pybedtools.BedTool(enhancer_file)])
        b = b.filter(lambda x: x.chrom in chroms)

        b = b.flank(g=g.sizes_file, l=1, r=0).sort().merge(d=1000, c=4, o="distinct")
        b.saveas(str(gene_file))

        logger.info("processing data files")
        # create coverage_table
        df = coverage_table(
            enhancer_file,
            data_files,
            window=window,
            log_transform=True,
            normalization="quantile",
            ncpus=12,
        )

        df.index.rename("loc", inplace=True)
        df.reset_index().to_feather(f"{outdir}/enhancers.feather")
        np.savez(target_file, target=df.iloc[:, 0].sort_values())
        meanstd = pd.DataFrame(
            index=df.index,
        )
        meanstd["mean"] = df.mean(1)
        meanstd["std"] = df.std(1)
        meanstd = meanstd.reset_index().rename(columns={"loc": "index"})

        meanstd.to_csv(meanstd_file, compression="gzip", index=False, sep="\t")
        df.index.rename("loc", inplace=True)
        df = df.sub(df.mean(1), axis=0)
        df = df.div(df.std(1), axis=0)
        df.reset_index().to_feather(f"{outdir}/enhancers.feather")

        link = create_link_file(meanstd_file, gene_file, genome=genome)
        link.to_feather(link_file)

        genes = _create_gene_table(
            df,
            meanstd_file,
            gene_file,
            gene_mapping,
            genome=genome,
            link_file=link_file,
            threshold=threshold
        )
        genes.to_csv(f"{outdir}/genes.txt", sep="\t")

        with open(f"{outdir}/info.yaml", "w") as f:
            yaml.dump(info, f)

        return ScepiaDataset(outdir)

    def extend(self, outdir: str, data_files: List[str]) -> T:

        if self.schema_version == "0.0.0":
            raise ValueError("dataset does not support custom sources")

        outdir = Path(outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        meanstd = pd.read_table(self.meanstd_file)
        bed = meanstd["index"].str.replace("[:-]", "\t").to_frame()

        logger.info("Processing BAM files")
        with NamedTemporaryFile() as f:
            bed.to_csv(f.name, index=False, header=False)

            # create coverage_table
            df = coverage_table(
                peakfile=f.name, datafiles=data_files, window=self.window, ncpus=12
            )
            target = np.load(self.target_file)["target"]
            df = qnorm.quantile_normalize(df, target=target)
            df.index = meanstd["index"]
            df = df.sub(meanstd["mean"].values, axis=0)
            df = df.div(meanstd["std"].values, axis=0)

        genes = _create_gene_table(
            df,
            self.meanstd_file,
            self.gene_file,
            self.gene_mapping,
            genome=self.genome,
            link_file=self.link_file,
        )
        logger.info(f"Writing reference to {outdir}")

        df.reset_index().to_feather(outdir / "enhancers.feather")
        genes.to_csv(outdir / "genes.txt", sep="\t")

        info = {
            "genes": "genes.txt",
            "enhancers": "enhancers.feather",
            "source": self.name,
            "genome": self.genome,
            "schema_version": __schema_version__,
        }

        with open(outdir / "info.yaml", "w") as f:
            yaml.dump(info, f)

        return ScepiaDataset(outdir)
