#!/bin/bash

# examples/helm/wmt_14/run.sh

# download med_dialog data to ./crfm-helm-public/lite/benchmark_output/runs/
./download.sh

# extract to .tsv
python extract.py

# run DKPS - generate plots
python plot_dkps.py

# use DKPS to predict model performance
python model_dkps.py