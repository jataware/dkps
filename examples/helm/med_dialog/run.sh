#!/bin/bash

# download med_dialog data to ./helm/medhelm/benchmark_output/runs/v2.0.0/
./download-med_dialog.sh

# extract to .tsv
python extract-med_dialog.py \
    --indir   ./helm/medhelm/benchmark_output/runs/v2.0.0/ \
    --outpath ./med_dialog.tsv

# run DKPS
python dkps-med_dialog.py