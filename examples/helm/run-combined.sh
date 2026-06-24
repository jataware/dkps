#!/bin/bash
# run-combined.sh — DKPS+IRT and APW→DKPS combined methods

set -e
source ~/.secrets

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Math
python run_combined.py --dataset "math:subject=algebra"                    --embed_provider google
python run_combined.py --dataset "math:subject=counting_and_probability"   --embed_provider google
python run_combined.py --dataset "math:subject=geometry"                   --embed_provider google
python run_combined.py --dataset "math:subject=intermediate_algebra"       --embed_provider google
python run_combined.py --dataset "math:subject=number_theory"              --embed_provider google
python run_combined.py --dataset "math:subject=prealgebra"                 --embed_provider google
python run_combined.py --dataset "math:subject=precalculus"                --embed_provider google

# LegalBench
python run_combined.py --dataset "legalbench:subset=abercrombie"                         --embed_model onehot
python run_combined.py --dataset "legalbench:subset=international_citizenship_questions" --embed_model onehot
python run_combined.py --dataset "legalbench:subset=corporate_lobbying"                  --embed_model onehot
python run_combined.py --dataset "legalbench:subset=function_of_decision_section"        --embed_model onehot
python run_combined.py --dataset "legalbench:subset=proa"                                --embed_model onehot

# MedQA
python run_combined.py --dataset "med_qa" --embed_model onehot

# WMT
python run_combined.py --dataset "wmt_14:language_pair=cs-en" --embed_provider google --score_col meteor --sample 0.2
python run_combined.py --dataset "wmt_14:language_pair=de-en" --embed_provider google --score_col meteor --sample 0.2
python run_combined.py --dataset "wmt_14:language_pair=fr-en" --embed_provider google --score_col meteor --sample 0.2
python run_combined.py --dataset "wmt_14:language_pair=hi-en" --embed_provider google --score_col meteor --sample 0.2
python run_combined.py --dataset "wmt_14:language_pair=ru-en" --embed_provider google --score_col meteor --sample 0.2



# --

python run_combined.py --dataset "math:subject=ALL"         --embed_provider google
python run_combined.py --dataset "wmt_14:language_pair=ALL" --embed_provider google --score_col meteor --sample 0.2
python run_combined.py --dataset "legalbench:subset=ALL"    --embed_model    onehot