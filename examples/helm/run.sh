#!/bin/bash

# run.sh

# --
# MATH

# bash download-scripts/download-math.sh

# extract to .tsv
# python extract.py --dataset math

MATH_SUBJECTS=(
    "math:subject=algebra"
    "math:subject=counting_and_probability"
    "math:subject=geometry"
    "math:subject=intermediate_algebra"
    "math:subject=number_theory"
    "math:subject=prealgebra"
    "math:subject=precalculus"
)

for subject in "${MATH_SUBJECTS[@]}"; do
    # python eda.py                 --dataset "$subject"
    # python run_dkps.py            --dataset "$subject" --runner dkps
    python plot_dkps.py           --dataset "$subject"
    
    # python run_dkps.py            --dataset "$subject" --runner qselect --n_replicates 1024
    # python plot_qselect.py        --dataset "$subject"
done

# # [SPECIAL] Embedding comparison for math:subject=counting_and_probability

EMBED_DATASET="math:subject=counting_and_probability"

EMBEDDINGS=(
    # "jina"
    # "google"
    # "litellm text-embedding-3-large"
    # "openrouter sentence-transformers/all-minilm-l6-v2"
    # "huggingface microsoft/codebert-base"
    "sentence-transformers nomic-ai/nomic-embed-text-v2-moe"
)

for embed_cfg in "${EMBEDDINGS[@]}"; do
    read -r provider model <<< "$embed_cfg"

    if [ -z "$model" ]; then
        outdir="results/embed-${provider}"
        model_args=()
    else
        model_safe="${model//\//_}"
        outdir="results/embed-${provider}-${model_safe}"
        model_args=(--embed_model "$model")
    fi

    echo "=== Embedding: provider=$provider model=${model:-default} ==="
    python run_dkps.py \
        --runner dkps \
        --dataset "$EMBED_DATASET" \
        --embed_provider "$provider" \
        "${model_args[@]}" \
        --outdir "$outdir"
done

python plot_dkps_compare_embeddings.py --dataset "$EMBED_DATASET"


# --
# WMT 14

# bash download-scripts/download-wmt_14.sh

# python extract.py --dataset wmt_14

WMT_PAIRS=(
    "wmt_14:language_pair=cs-en"
    "wmt_14:language_pair=de-en"
    "wmt_14:language_pair=fr-en"
    "wmt_14:language_pair=hi-en"
    "wmt_14:language_pair=ru-en"
)

for pair in "${WMT_PAIRS[@]}"; do
    python eda.py                 --dataset "$pair" --sample 0.2
    # python run_dkps.py            --dataset "$pair" --score_col meteor --sample 0.2 --runner dkps
    # python plot_dkps.py           --dataset "$pair" --score_col meteor
    
    # python run_dkps.py            --dataset "$pair" --score_col meteor --sample 0.2 --runner qselect --n_replicates 1024
    # python plot_qselect.py        --dataset "$pair" --score_col meteor
done

# --
# LegalBench

# bash download-scripts/download-legalbench.sh

# python extract.py --dataset legalbench

LEGALBENCH_SUBSETS=(
    "legalbench:subset=abercrombie"
    "legalbench:subset=international_citizenship_questions"
    "legalbench:subset=corporate_lobbying"
    "legalbench:subset=function_of_decision_section"
    "legalbench:subset=proa"
)

for subset in "${LEGALBENCH_SUBSETS[@]}"; do
    python eda.py                 --dataset "$subset" --embed_model onehot
    # python run_dkps.py            --dataset "$subset" --embed_model onehot --runner dkps
    # python plot_dkps.py           --dataset "$subset"
    
    # python run_dkps.py            --dataset "$subset" --embed_model onehot --runner qselect --n_replicates 1024
    # python plot_qselect.py        --dataset "$subset"
done

# --
# MEDQA

# bash download-scripts/download-med_qa.sh

# python extract.py             --dataset med_qa

python eda.py                 --dataset med_qa --embed_model onehot
# python run_dkps.py            --dataset med_qa --embed_model onehot --runner dkps
# python plot_dkps.py           --dataset med_qa

# python run_dkps.py            --dataset med_qa --embed_model onehot --runner qselect --n_replicates 1024
# python plot_qselect.py        --dataset med_qa
