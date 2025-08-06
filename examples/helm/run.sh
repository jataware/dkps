#!/bin/bash

# examples/helm/wmt_14/run.sh

# --
# MATH

bash download-math.sh

# extract to .tsv
python extract.py --dataset math

# run DKPS - generate plots
# python plot_dkps.py --dataset math:subject=algebra
# python plot_dkps.py --dataset math:subject=counting_and_probability
# python plot_dkps.py --dataset math:subject=geometry
python plot_dkps.py --dataset math:subject=intermediate_algebra
# python plot_dkps.py --dataset math:subject=number_theory
# python plot_dkps.py --dataset math:subject=prealgebra
# python plot_dkps.py --dataset math:subject=precalculus

# use DKPS to predict model performance
# python model_dkps.py --dataset math:subject=algebra
# python model_dkps.py --dataset math:subject=counting_and_probability
# python model_dkps.py --dataset math:subject=geometry
python model_dkps.py --dataset math:subject=intermediate_algebra
# python model_dkps.py --dataset math:subject=number_theory
# python model_dkps.py --dataset math:subject=prealgebra
# python model_dkps.py --dataset math:subject=precalculus

# analyze DKPS model performance results
# python model_dkps_analysis.py --dataset math:subject=algebra
# python model_dkps_analysis.py --dataset math:subject=counting_and_probability
# python model_dkps_analysis.py --dataset math:subject=geometry
# python model_dkps_analysis.py --dataset math:subject=intermediate_algebra
python model_dkps_analysis.py --dataset math:subject=number_theory
# python model_dkps_analysis.py --dataset math:subject=prealgebra
# python model_dkps_analysis.py --dataset math:subject=precalculus

# --
# WMT 14

bash download-wmt_14.sh

python extract.py --dataset wmt_14

# run DKPS - generate plots
python plot_dkps.py --dataset wmt_14:language_pair=cs-en
# python plot_dkps.py --dataset wmt_14:language_pair=de-en
# python plot_dkps.py --dataset wmt_14:language_pair=fr-en
# python plot_dkps.py --dataset wmt_14:language_pair=hi-en
# python plot_dkps.py --dataset wmt_14:language_pair=ru-en

# use DKPS to predict model performance
python model_dkps.py --dataset wmt_14:language_pair=cs-en --score_col meteor
# python model_dkps.py --dataset wmt_14:language_pair=de-en --score_col meteor
# python model_dkps.py --dataset wmt_14:language_pair=fr-en --score_col meteor
# python model_dkps.py --dataset wmt_14:language_pair=hi-en --score_col meteor
# python model_dkps.py --dataset wmt_14:language_pair=ru-en --score_col meteor

python model_dkps_analysis.py --dataset wmt_14:language_pair=cs-en --score_col meteor
# python model_dkps_analysis.py --dataset wmt_14:language_pair=de-en --score_col meteor
# python model_dkps_analysis.py --dataset wmt_14:language_pair=fr-en --score_col meteor
# python model_dkps_analysis.py --dataset wmt_14:language_pair=hi-en --score_col meteor
# python model_dkps_analysis.py --dataset wmt_14:language_pair=ru-en --score_col meteor
