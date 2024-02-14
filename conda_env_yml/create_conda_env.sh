#!/usr/bin/env bash


# settings to catch errors in bash scripts
set -euf -o pipefail

module load Anaconda3/2022.05

my_conda_storage=/nird/home/johannef/.conda


export CONDA_PKGS_DIRS=${my_conda_storage}/package-cache
conda env create --prefix ${my_conda_storage}/msc_env --file msc_env.yml