#!/bin/bash

# examples/helm/wmt_14/run.sh

# download med_dialog data to ./crfm-helm-public/lite/benchmark_output/runs/
./download.sh

# extract to .tsv
python extract.py

# run DKPS
python run_dkps.py