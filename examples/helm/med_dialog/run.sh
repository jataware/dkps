#!/bin/bash

# examples/helm/med_dialog/run.sh

# download med_dialog data to ./helm/medhelm/benchmark_output/runs/v2.0.0/
./download.sh

# extract to .tsv
python extract.py \
    --indir   ./helm/medhelm/benchmark_output/runs/v2.0.0/ \
    --outpath ./med_dialog.tsv

# run DKPS
python run_dkps.py