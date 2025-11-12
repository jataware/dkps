#!/bin/bash

# examples/helm/wmt_14/run.sh

# --
# MATH

bash download-scripts/download-math.sh

# extract to .tsv
python extract.py --dataset math

# run DKPS - generate plots
python plot_dkps.py --dataset math:subject=algebra
python plot_dkps.py --dataset math:subject=counting_and_probability
python plot_dkps.py --dataset math:subject=geometry
python plot_dkps.py --dataset math:subject=intermediate_algebra
python plot_dkps.py --dataset math:subject=number_theory
python plot_dkps.py --dataset math:subject=prealgebra
python plot_dkps.py --dataset math:subject=precalculus

# use DKPS to predict model performance
python model_dkps.py --dataset math:subject=algebra
python model_dkps.py --dataset math:subject=counting_and_probability
python model_dkps.py --dataset math:subject=geometry
python model_dkps.py --dataset math:subject=intermediate_algebra
python model_dkps.py --dataset math:subject=number_theory
python model_dkps.py --dataset math:subject=prealgebra
python model_dkps.py --dataset math:subject=precalculus

# plot DKPS model performance results
python model_dkps_analysis.py --dataset math:subject=algebra
python model_dkps_analysis.py --dataset math:subject=counting_and_probability
python model_dkps_analysis.py --dataset math:subject=geometry
python model_dkps_analysis.py --dataset math:subject=intermediate_algebra
python model_dkps_analysis.py --dataset math:subject=number_theory
python model_dkps_analysis.py --dataset math:subject=prealgebra
python model_dkps_analysis.py --dataset math:subject=precalculus

# --
# WMT 14

bash download-scripts/download-wmt_14.sh

python extract.py --dataset wmt_14

# run DKPS - generate plots
python plot_dkps.py --dataset wmt_14:language_pair=cs-en --sample 0.2
python plot_dkps.py --dataset wmt_14:language_pair=de-en --sample 0.2
python plot_dkps.py --dataset wmt_14:language_pair=fr-en --sample 0.2
python plot_dkps.py --dataset wmt_14:language_pair=hi-en --sample 0.2
python plot_dkps.py --dataset wmt_14:language_pair=ru-en --sample 0.2

# use DKPS to predict model performance
python model_dkps.py --dataset wmt_14:language_pair=cs-en --score_col meteor --sample 0.2
python model_dkps.py --dataset wmt_14:language_pair=de-en --score_col meteor --sample 0.2
python model_dkps.py --dataset wmt_14:language_pair=fr-en --score_col meteor --sample 0.2
python model_dkps.py --dataset wmt_14:language_pair=hi-en --score_col meteor --sample 0.2
python model_dkps.py --dataset wmt_14:language_pair=ru-en --score_col meteor --sample 0.2

# plot DKPS model performance results
python model_dkps_analysis.py --dataset wmt_14:language_pair=cs-en --score_col meteor
python model_dkps_analysis.py --dataset wmt_14:language_pair=de-en --score_col meteor
python model_dkps_analysis.py --dataset wmt_14:language_pair=fr-en --score_col meteor
python model_dkps_analysis.py --dataset wmt_14:language_pair=hi-en --score_col meteor
python model_dkps_analysis.py --dataset wmt_14:language_pair=ru-en --score_col meteor

# --
# MEDQA

bash download-scripts/download-med_qa.sh

python extract.py             --dataset med_qa

python plot_dkps.py           --dataset med_qa --embed_model onehot
python model_dkps.py          --dataset med_qa --embed_model onehot
python model_dkps_analysis.py --dataset med_qa

# --
# LegalBench

bash download-scripts/download-legalbench.sh

python extract.py --dataset legalbench

python plot_dkps.py --dataset legalbench:subset=abercrombie                         --embed_model onehot
python plot_dkps.py --dataset legalbench:subset=international_citizenship_questions --embed_model onehot
python plot_dkps.py --dataset legalbench:subset=corporate_lobbying                  --embed_model onehot
python plot_dkps.py --dataset legalbench:subset=function_of_decision_section        --embed_model onehot
python plot_dkps.py --dataset legalbench:subset=proa                                --embed_model onehot

python model_dkps.py --dataset legalbench:subset=abercrombie                         --embed_model onehot
python model_dkps.py --dataset legalbench:subset=international_citizenship_questions --embed_model onehot
python model_dkps.py --dataset legalbench:subset=corporate_lobbying                  --embed_model onehot
python model_dkps.py --dataset legalbench:subset=function_of_decision_section        --embed_model onehot
python model_dkps.py --dataset legalbench:subset=proa                                --embed_model onehot

python model_dkps_analysis.py --dataset legalbench:subset=abercrombie
python model_dkps_analysis.py --dataset legalbench:subset=international_citizenship_questions
python model_dkps_analysis.py --dataset legalbench:subset=corporate_lobbying
python model_dkps_analysis.py --dataset legalbench:subset=function_of_decision_section
python model_dkps_analysis.py --dataset legalbench:subset=proa




# <<<
python model_dkps.py --dataset math:subject=algebra --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset math:subject=counting_and_probability --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset math:subject=geometry --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset math:subject=intermediate_algebra --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset math:subject=number_theory --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset math:subject=prealgebra --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset math:subject=precalculus --n_replicates 512 --n_jobs -1

python model_dkps.py --dataset med_qa --embed_model onehot --n_replicates 512 --n_jobs -1

python model_dkps.py --dataset legalbench:subset=abercrombie                         --embed_model onehot --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset legalbench:subset=international_citizenship_questions --embed_model onehot --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset legalbench:subset=corporate_lobbying                  --embed_model onehot --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset legalbench:subset=function_of_decision_section        --embed_model onehot --n_replicates 512 --n_jobs -1
python model_dkps.py --dataset legalbench:subset=proa                                --embed_model onehot --n_replicates 512 --n_jobs -1
# >>>