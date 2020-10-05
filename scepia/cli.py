#!/usr/bin/env python
import click
import scepia
import sys
import os
from scepia import _version
from scepia import generate_signal, link_it_up
from scepia.sc import full_analysis

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(_version)
def cli():
    """SCEPIA - Single Cell Epigenome-based Inference of Activity

    Version: {}""".format(
        _version
    )
    pass


@click.command("area27", short_help="Determine the enhancer-based regulatory potential (ERP) score.")
@click.argument("bamfile")
@click.argument("outfile")
@click.option("-w", "--window", help="window")
@click.option("-N", "--nthreads", help="number of threads")
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
@click.argument("outfile")
def infer_motifs(infile, outfile):
    """
    Infer motifs.
    """
    full_analysis(infile, outfile)   


cli.add_command(area27)
cli.add_command(infer_motifs)

if __name__ == "__main__":
    cli()
