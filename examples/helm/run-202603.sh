#!/bin/bash

# run.sh

set -e
source ~/.secrets

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# --
# MATH

# bash download-scripts/download-math.sh

# extract to .tsv
# python extract.py --dataset math

MATH_SUBJECTS=(
    "math:subject=algebra"
    "math:subject=counting_and_probability"
    "math:subject=geometry"
    "math:subject=intermediate_algebra"
    "math:subject=number_theory"
    "math:subject=prealgebra"
    "math:subject=precalculus"
)

for embed_provider in "google"; do
    for subject in "${MATH_SUBJECTS[@]}"; do
        # # python eda.py                 --dataset "$subject"
        python run_dkps.py            --dataset "$subject" --runner dkps --n_replicates 100 --embed_provider "$embed_provider"
        # python plot_dkps.py           --dataset "$subject" --embed_provider "$embed_provider"        
        # python run_dkps.py            --dataset "$subject" --runner qselect --n_replicates 1024 --embed_provider "$embed_provider"
        # python plot_qselect.py        --dataset "$subject" --embed_provider "$embed_provider"
    done
done


# # --
# # WMT 14

# # bash download-scripts/download-wmt_14.sh

# # python extract.py --dataset wmt_14



WMT_PAIRS=(
    "wmt_14:language_pair=cs-en"
    "wmt_14:language_pair=de-en"
    "wmt_14:language_pair=fr-en"
    "wmt_14:language_pair=hi-en"
    "wmt_14:language_pair=ru-en"
)

for embed_provider in "google"; do
    for pair in "${WMT_PAIRS[@]}"; do
        # python eda.py                 --dataset "$pair" --sample 0.2
        python run_dkps.py            --dataset "$pair" --score_col meteor --sample 0.2 --runner dkps --n_replicates 100 --embed_provider "$embed_provider"
        # python plot_dkps.py           --dataset "$pair" --score_col meteor --embed_provider "$embed_provider"
        # python run_dkps.py            --dataset "$pair" --score_col meteor --sample 0.2 --runner qselect --n_replicates 1024 --embed_provider "$embed_provider"
        # python plot_qselect.py        --dataset "$pair" --score_col meteor --embed_provider "$embed_provider"
    done
done


# --
# LegalBench

# bash download-scripts/download-legalbench.sh

# python extract.py --dataset legalbench

LEGALBENCH_SUBSETS=(
    "legalbench:subset=abercrombie"
    "legalbench:subset=international_citizenship_questions"
    "legalbench:subset=corporate_lobbying"
    "legalbench:subset=function_of_decision_section"
    "legalbench:subset=proa"
)

for subset in "${LEGALBENCH_SUBSETS[@]}"; do
    # python eda.py                 --dataset "$subset" --embed_model onehot
    python run_dkps.py            --dataset "$subset" --embed_model onehot --runner dkps --n_replicates 100
    # python plot_dkps.py           --dataset "$subset" --embed_model onehot
    
    # python run_dkps.py            --dataset "$subset" --embed_model onehot --runner qselect --n_replicates 1024
    # python plot_qselect.py        --dataset "$subset" --embed_model onehot
done


# # --
# # MEDQA

# # bash download-scripts/download-med_qa.sh

# # python extract.py             --dataset med_qa

# # python eda.py                 --dataset med_qa --embed_model onehot
python run_dkps.py            --dataset med_qa --embed_model onehot --runner dkps --n_replicates 100
# python plot_dkps.py           --dataset med_qa --embed_model onehot

# # python run_dkps.py            --dataset med_qa --embed_model onehot --runner qselect --n_replicates 1024
# # python plot_qselect.py        --dataset med_qa --embed_model onehot


# # --
# # Make tables

python make_table.py --embed_model    onehot
python make_table.py --embed_provider google
# python make_table.py --embed_provider jina

# python make_table_qselect.py --embed_model    onehot
# python make_table_qselect.py --embed_provider google
# python make_table_qselect.py --embed_provider jina

# # --
# # Print model metadata

# python print_models.py
