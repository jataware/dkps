#!/usr/bin/env bash
set -euo pipefail

PRESET="quick"
OUTDIR="./results/${PRESET}"

python -m experiments.unpaired.run_block1 --experiment s0 --preset "${PRESET}" --output-dir "${OUTDIR}/s0"
python -m experiments.unpaired.run_block1 --experiment s1 --preset "${PRESET}" --output-dir "${OUTDIR}/s1"
python -m experiments.unpaired.run_block1 --experiment s2 --preset "${PRESET}" --output-dir "${OUTDIR}/s2"
python -m experiments.unpaired.run_block1 --experiment s3 --preset "${PRESET}" --output-dir "${OUTDIR}/s3"
