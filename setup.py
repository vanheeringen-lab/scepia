import versioneer
from setuptools import setup, find_packages
import os

DESCRIPTION = (
    "Inference of transcription factor motif activity from single cell RNA-seq data."
)

with open("README.md") as f:
    long_description = f.read()

entry_points = {"console_scripts": ["scepia=scepia.cli:cli"]}


setup(
    name="scepia",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    description=DESCRIPTION,
    author="Simon van Heeringen",
    author_email="simon.vanheeringen@gmail.com",
    url="https://github.com/vanheeringen-lab/scepia/",
    license="MIT",
    packages=find_packages(),
    entry_points=entry_points,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    install_requires=[
        "adjustText",
        "biofluff >=3.0.4",
        "geosketch",
        "gimmemotifs >=0.15.2",
        "leidenalg",
        "loguru",
        "matplotlib >=3.7.2",
        "numba",
        "numpy",
        "pandas",
        "scanpy",
        "scikit-learn",
        "scipy",
        "seaborn",
        "tqdm",
        "xdg",
    ],
)
