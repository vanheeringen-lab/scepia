# Copyright (c) 2020 Simon van Heeringen <simon.vanheeringen@gmail.com>
#
# This module is free software. You can redistribute it and/or modify it under
# the terms of the MIT License, see the file LICENSE included with this
# distribution.
import inspect
import os
import shutil
from typing import Optional
import tarfile
import urllib

from appdirs import user_cache_dir
from loguru import logger
from numba import njit
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(user_cache_dir("scepia"))


@njit
def mean1(a):
    n = len(a)
    b = np.empty(n)
    for i in range(n):
        b[i] = a[i].mean()
    return b


@njit
def std1(a):
    n = len(a)
    b = np.empty(n)
    for i in range(n):
        b[i] = a[i].std()
    return b


@njit
def fast_corr(a, b):
    """ Correlation """
    n, k = a.shape
    m, k = b.shape

    mu_a = mean1(a)
    mu_b = mean1(b)
    sig_a = std1(a)
    sig_b = std1(b)

    out = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            out[i, j] = (a[i] - mu_a[i]) @ (b[j] - mu_b[j]) / k / sig_a[i] / sig_b[j]

    return out


def locate_data(dataset: str, version: Optional[float] = None) -> str:
    """Locate reference data for cell type annotation.

    Data set will be downloaded if it can not be found.

    Parameters
    ----------
    dataset : `str`
        Name of dataset. Can be a local directory.
    version : `float`, optional
        Version of dataset

    Returns
    -------
    `str`
        Absolute path to data set directory.
    """
    for datadir in [os.path.expanduser(dataset), os.path.join(CACHE_DIR, dataset)]:
        if os.path.isdir(datadir):
            if os.path.exists(os.path.join(datadir, "info.yaml")):
                return datadir
            else:
                raise ValueError(f"info.yaml not found in directory {datadir}")

    # Data file can not be found
    # Read the data directory from the installed module
    # Once the github is public, it can be read from the github repo directly
    install_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    df = pd.read_csv(os.path.join(install_dir, "../data/data_directory.txt"), sep="\t")
    df = df[df["name"] == dataset]
    if df.shape[0] > 0:
        if version is None:
            url = df.sort_values("version").tail(1)["url"].values[0]
        else:
            url_df = df.loc[df["version"] == version]
            if url_df.shape[0] > 0:
                url = url_df["url"].values[0]
            else:
                raise ValueError(f"Dataset {dataset} with version {version} not found.")
        datadir = os.path.join(CACHE_DIR, dataset)
        os.mkdir(datadir)
        datafile = os.path.join(datadir, os.path.split(url)[-1])
        logger.info(f"Downloading {dataset} data files to {datadir}...\n")
        with urllib.request.urlopen(url) as response, open(datafile, "wb") as outfile:
            shutil.copyfileobj(response, outfile)

        logger.info("Extracting files...\n")
        tf = tarfile.open(datafile)
        tf.extractall(datadir)
        os.unlink(datafile)

        return datadir
    else:
        raise ValueError(f"Dataset {dataset} not found.")
