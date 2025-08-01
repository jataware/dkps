#!/bin/bash

# download.sh

# download scenario_state.json
gcloud storage ls "gs://crfm-helm-public/lite/benchmark_output/runs/*/wmt_14:language_pair=de-en*/scenario_state.json" > urls1.txt
cat urls1.txt | sed 's@gs://@https://storage.googleapis.com/@' > tmp && mv tmp urls1.txt
wget -x -nH -i urls1.txt
rm urls1.txt

# download scenario_state.json
gcloud storage ls "gs://crfm-helm-public/lite/benchmark_output/runs/*/wmt_14:language_pair=de-en*/display_predictions.json" > urls2.txt
cat urls2.txt | sed 's@gs://@https://storage.googleapis.com/@' > tmp && mv tmp urls2.txt
wget -x -nH -i urls2.txt
rm urls2.txt

# remove extra run?
rm -r ./crfm-helm-public/lite/benchmark_output/runs/v1.8.0-nemotron/