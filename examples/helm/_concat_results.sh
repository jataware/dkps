#!/bin/bash
# Concatenate results.tsv and results-1.tsv files, saving to results-v2.tsv

set -e

RESULTS_DIR="${1:-results}"

find "$RESULTS_DIR" -name "results.tsv" | while read -r results_file; do
    dir=$(dirname "$results_file")
    results1_file="$dir/results-1.tsv"
    output_file="$dir/results-v2.tsv"

    if [[ -f "$results1_file" ]]; then
        echo "Concatenating: $dir"
        # Copy results.tsv (with header) and append results-1.tsv (without header)
        cat "$results_file" > "$output_file"
        tail -n +2 "$results1_file" >> "$output_file"
        echo "  -> $output_file"
    else
        echo "Skipping $dir (no results-1.tsv)"
    fi
done

echo "Done!"
