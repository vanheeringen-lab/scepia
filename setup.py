from setuptools import setup, find_packages

import os
import glob
import sys
from io import open

CONFIG_NAME = "gimmemotifs.cfg" 
DESCRIPTION  = "GimmeMotifs is a motif prediction pipeline."

#with open('README.md', encoding='utf-8') as f:
#    long_description = f.read().strip("\n")

# are we in the conda build environment?
conda_build = os.environ.get("CONDA_BUILD")

setup (
        name = 'area27',
        version = '0.1.0',
#        long_description = long_description,
#        long_description_content_type = 'text/markdown',
        description = DESCRIPTION,
        author = 'Simon van Heeringen',
        author_email = 'simon.vanheeringen@gmail.com',
        url = 'https://github.com/simonvh/gimmemotifs/',
#        download_url = 'https://github.com/simonvh/gimmemotifs/tarball/' + versioneer.get_version(),
        license = 'MIT',
        packages=find_packages(),
        scripts=[
            'scripts/area27',
            ],
        include_package_data = True,
        zip_safe = False,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            ],
        install_requires = [
            "pandas",
            "biofluff",
            "xdg",
        ],
)
