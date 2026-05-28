#!/bin/bash

# run.sh

set -e
source ~/.secrets

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

embed_provider='google'

# --------------------------------------------------
# MATH

# <<
# Whole dataset
# TODO
python run_dkps.py            --outdir results-all --dataset "math:subject=ALL" --runner dkps --n_replicates 1024 --embed_provider "$embed_provider"
# python plot_dkps.py           --outdir results-all --dataset "math:subject=ALL" --embed_provider "$embed_provider"        
python run_dkps.py            --outdir results-all --dataset "math:subject=ALL" --runner qselect --n_replicates 1024 --embed_provider "$embed_provider"
# python plot_qselect.py        --outdir results-all --dataset "math:subject=ALL" --embed_provider "$embed_provider"


# --------------------------------------------------
# WMT 14

python run_dkps.py            --outdir results-all --dataset "wmt_14:language_pair=ALL" --score_col meteor --sample 0.2 --runner dkps --n_replicates 1024 --embed_provider "$embed_provider"
# python plot_dkps.py           --outdir results-all --dataset "wmt_14:language_pair=ALL" --score_col meteor --embed_provider "$embed_provider"
python run_dkps.py            --outdir results-all --dataset "wmt_14:language_pair=ALL" --score_col meteor --sample 0.2 --runner qselect --n_replicates 1024 --embed_provider "$embed_provider"
# python plot_qselect.py        --outdir results-all --dataset "wmt_14:language_pair=ALL" --score_col meteor --embed_provider "$embed_provider"


# --------------------------------------------------
# LegalBench

python run_dkps.py            --outdir results-all --dataset "legalbench:subset=ALL" --embed_model onehot --runner dkps --n_replicates 1024
# python plot_dkps.py           --outdir results-all --dataset "legalbench:subset=ALL" --embed_model onehot
python run_dkps.py            --outdir results-all --dataset "legalbench:subset=ALL" --embed_model onehot --runner qselect --n_replicates 1024
# python plot_qselect.py        --outdir results-all --dataset "legalbench:subset=ALL" --embed_model onehot


# --------------------------------------------------
# MEDQA

# ONLY ONE SPLIT


# # --------------------------------------------------
# # Make tables - TODO

python make_table.py --embed_model    onehot --results_dir results-all
python make_table.py --embed_provider google --results_dir results-all
# python make_table.py --embed_provider jina   --results_dir results-all

python make_table_qselect.py --embed_model    onehot --results_dir results-all
python make_table_qselect.py --embed_provider google --results_dir results-all
# python make_table_qselect.py --embed_provider jina

# # --
# # Print model metadata

# python print_models.py
