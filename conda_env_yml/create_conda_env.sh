#!/usr/bin/env bash


# settings to catch errors in bash scripts
set -euf -o pipefail

module load Anaconda3/2022.05

my_conda_storage=/nird/home/johannef/.conda
new_env_name=msc_env
yml_file=/nird/home/johannef/Masterthesis_S23/conda_env_yml/msc_env.yml

# Remove the existing prefix directory if it exists
if [ -d "$my_conda_storage/$new_env_name" ]; then
    echo "Removing existing Conda environment $new_env_name directory..."
    rm -rf $my_conda_storage/$new_env_name
fi

export CONDA_PKGS_DIRS=${my_conda_storage}/package-cache
conda env create --prefix ${my_conda_storage}/${new_env_name} --file ${yml_file}