#!/bin/bash
# Shore-up + conditional analysis: re-run every suite result at 16 seeds, add pairing-cliff
# budget robustness (2,4,8), and dump per-(model,task) errors for the conditional analysis.
set -e
cd /home/paperspace/projects/dkps/examples/helm
SECONDS=0
step () { echo; echo "===== $1 ($(date +%H:%M:%S), +${SECONDS}s) ====="; }

step "RD1 pairing cliff -- robustness over budgets {2,4,8}"
pixi run python helm_rd1_suite.py --sweep pairing --pairing_budgets 2 4 8 --n_seeds 16

step "RD1 query efficiency -- budget"
pixi run python helm_rd1_suite.py --sweep budget --n_seeds 16
step "RD1 query efficiency -- n_models"
pixi run python helm_rd1_suite.py --sweep n_models --n_seeds 16
step "RD1 query efficiency -- coverage"
pixi run python helm_rd1_suite.py --sweep coverage --n_seeds 16

step "RD1 breakdown -- per-(model,task) errors at unpaired/paired (conditional analysis)"
pixi run python helm_rd1_suite.py --sweep breakdown --pairing_budget 4 --n_paired_values 0 4 --n_seeds 16

step "Completion -- coverage"
pixi run python helm_completion_suite.py --sweep coverage --n_seeds 16
step "Completion -- p_query"
pixi run python helm_completion_suite.py --sweep p_query --n_seeds 16
step "Completion -- n_tasks"
pixi run python helm_completion_suite.py --sweep n_tasks --n_seeds 16
step "Completion -- n_models"
pixi run python helm_completion_suite.py --sweep n_models --n_seeds 16

echo; echo "===== ALL DONE (+${SECONDS}s) ====="
