#!/usr/bin/env python
import click
from scepia import __version__
from scepia import generate_signal, link_it_up
from scepia.sc import full_analysis

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def cli():
    """SCEPIA - Single Cell Epigenome-based Inference of Activity

    Version: {}""".format(
        __version__
    )
    pass


@click.command(
    "area27",
    short_help="Determine the enhancer-based regulatory potential (ERP) score.",
)
@click.argument("bamfile")
@click.argument("outfile")
@click.option("-w", "--window", help="window")
@click.option("-N", "--nthreads", help="Number of threads.")
def area27(bamfile, outfile, window=2000, nthreads=4):
    """
    Determine the enhancer-based regulatory potential (ERP) score per gene. This
    approach is based on the method developed by Wang et al., 2016. There is one
    difference. In this implementation the score is calculated based only on
    H3K27ac signal in enhancers. We use log-transformed, z-score normalized
    H3K27ac read counts in 2kb windows centered at enhancer locations.
    """
    signal = generate_signal(bamfile, window=2000, nthreads=nthreads)
    link_it_up(outfile, signal)


@click.command("infer_motifs", short_help="Run SCEPIA motif inference on an h5ad file.")
@click.argument("infile")
@click.argument("outdir")
@click.option(
    "-c", "--cluster", help="cluster name (default checks for 'louvain' or 'leiden')."
)
@click.option(
    "-n",
    "--n_top_genes",
    default=1000,
    help="Maximum number of variable genes that is used (1000).",
)
@click.option(
    "-p", "--pfmfile", help="Name of motif PFM file or GimmeMotifs database name."
)
@click.option(
    "-a",
    "--min_annotated",
    default=50,
    help="Minimum number of cells per cell type (50).",
)
@click.option(
    "-e",
    "--num_enhancers",
    default=10000,
    help="Number of enhancers to use for motif activity (10000).",
)
def infer_motifs(
    infile, outdir, cluster, n_top_genes, pfmfile, min_annotated, num_enhancers
):
    """
    Infer motifs.
    """
    full_analysis(
        infile,
        outdir,
        cluster=cluster,
        n_top_genes=n_top_genes,
        pfmfile=pfmfile,
        min_annotated=min_annotated,
        num_enhancers=num_enhancers,
    )


cli.add_command(area27)
cli.add_command(infer_motifs)

if __name__ == "__main__":
    cli()
