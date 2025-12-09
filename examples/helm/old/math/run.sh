#!/bin/bash

# examples/helm/wmt_14/run.sh

# download med_dialog data to ./crfm-helm-public/lite/benchmark_output/runs/
./download.sh

# extract to .tsv
python extract.py

# run DKPS - generate plots
python plot_dkps.py --dataset 'math:subject=algebra'
python plot_dkps.py --dataset 'math:subject=counting_and_probability'
python plot_dkps.py --dataset 'math:subject=geometry'
python plot_dkps.py --dataset 'math:subject=intermediate_algebra'
python plot_dkps.py --dataset 'math:subject=number_theory'
python plot_dkps.py --dataset 'math:subject=prealgebra'
python plot_dkps.py --dataset 'math:subject=precalculus'

# use DKPS to predict model performance
python model_dkps.py --dataset 'math:subject=algebra'
python model_dkps.py --dataset 'math:subject=counting_and_probability'
python model_dkps.py --dataset 'math:subject=geometry'
python model_dkps.py --dataset 'math:subject=intermediate_algebra'
python model_dkps.py --dataset 'math:subject=number_theory'
python model_dkps.py --dataset 'math:subject=prealgebra'
python model_dkps.py --dataset 'math:subject=precalculus'
