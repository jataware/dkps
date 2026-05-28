#!/bin/bash
# run-baselines.sh — IRT + APW baselines for rebuttal Table 1

set -e

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Math
python run_baselines.py --dataset "math:subject=counting_and_probability"
python run_baselines.py --dataset "math:subject=algebra"
python run_baselines.py --dataset "math:subject=geometry"
python run_baselines.py --dataset "math:subject=intermediate_algebra"
python run_baselines.py --dataset "math:subject=number_theory"
python run_baselines.py --dataset "math:subject=prealgebra"
python run_baselines.py --dataset "math:subject=precalculus"

# LegalBench
python run_baselines.py --dataset "legalbench:subset=abercrombie"
python run_baselines.py --dataset "legalbench:subset=international_citizenship_questions"
python run_baselines.py --dataset "legalbench:subset=corporate_lobbying"
python run_baselines.py --dataset "legalbench:subset=function_of_decision_section"
python run_baselines.py --dataset "legalbench:subset=proa"

# WMT
python run_baselines.py --dataset "wmt_14:language_pair=cs-en" --score_col meteor --sample 0.2
python run_baselines.py --dataset "wmt_14:language_pair=de-en" --score_col meteor --sample 0.2
python run_baselines.py --dataset "wmt_14:language_pair=fr-en" --score_col meteor --sample 0.2
python run_baselines.py --dataset "wmt_14:language_pair=hi-en" --score_col meteor --sample 0.2
                                                                                                                                    
# MedQA
python run_baselines.py --dataset "med_qa" --n_lofo 10