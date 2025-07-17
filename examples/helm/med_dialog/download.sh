#!/bin/bash

# download-med_dialog.sh

# ... ideally we could do this through gcloud, but these most recent runs are missing ... check back later

RUNS=(
    "med_dialog,subset=healthcaremagic:model=anthropic_claude-3-5-sonnet-20241022,model_deployment=stanfordhealthcare_claude-3-5-sonnet-20241022"
    "med_dialog,subset=healthcaremagic:model=anthropic_claude-3-7-sonnet-20250219,model_deployment=stanfordhealthcare_claude-3-7-sonnet-20250219"
    "med_dialog,subset=healthcaremagic:model=google_gemini-1.5-pro-001,model_deployment=stanfordhealthcare_gemini-1.5-pro-001"
    "med_dialog,subset=healthcaremagic:model=google_gemini-2.0-flash-001,model_deployment=stanfordhealthcare_gemini-2.0-flash-001"
    "med_dialog,subset=healthcaremagic:model=meta_llama-3.3-70b-instruct,model_deployment=stanfordhealthcare_llama-3.3-70b-instruct"
    "med_dialog,subset=healthcaremagic:model=openai_gpt-4o-2024-05-13,model_deployment=stanfordhealthcare_gpt-4o-2024-05-13"
    "med_dialog,subset=healthcaremagic:model=openai_gpt-4o-mini-2024-07-18,model_deployment=stanfordhealthcare_gpt-4o-mini-2024-07-18"
    "med_dialog,subset=healthcaremagic:num_output_tokens=4000,model=deepseek-ai_deepseek-r1,model_deployment=stanfordhealthcare_deepseek-r1"
    "med_dialog,subset=healthcaremagic:num_output_tokens=4000,model=openai_o3-mini-2025-01-31,model_deployment=stanfordhealthcare_o3-mini-2025-01-31"
    "med_dialog,subset=icliniq:model=anthropic_claude-3-5-sonnet-20241022,model_deployment=stanfordhealthcare_claude-3-5-sonnet-20241022"
    "med_dialog,subset=icliniq:model=anthropic_claude-3-7-sonnet-20250219,model_deployment=stanfordhealthcare_claude-3-7-sonnet-20250219"
    "med_dialog,subset=icliniq:model=google_gemini-1.5-pro-001,model_deployment=stanfordhealthcare_gemini-1.5-pro-001"
    "med_dialog,subset=icliniq:model=google_gemini-2.0-flash-001,model_deployment=stanfordhealthcare_gemini-2.0-flash-001"
    "med_dialog,subset=icliniq:model=meta_llama-3.3-70b-instruct,model_deployment=stanfordhealthcare_llama-3.3-70b-instruct"
    "med_dialog,subset=icliniq:model=openai_gpt-4o-2024-05-13,model_deployment=stanfordhealthcare_gpt-4o-2024-05-13"
    "med_dialog,subset=icliniq:model=openai_gpt-4o-mini-2024-07-18,model_deployment=stanfordhealthcare_gpt-4o-mini-2024-07-18"
    "med_dialog,subset=icliniq:num_output_tokens=4000,model=deepseek-ai_deepseek-r1,model_deployment=stanfordhealthcare_deepseek-r1"
    "med_dialog,subset=icliniq:num_output_tokens=4000,model=openai_o3-mini-2025-01-31,model_deployment=stanfordhealthcare_o3-mini-2025-01-31"
)

echo "" > urls.txt
BASE_URL="https://nlp.stanford.edu/helm/medhelm/benchmark_output/runs/v2.0.0"
for run in "${RUNS[@]}"; do
    echo ${BASE_URL}/${run}/display_predictions.json  >> urls.txt
    echo ${BASE_URL}/${run}/display_requests.json     >> urls.txt
    echo ${BASE_URL}/${run}/instances.json            >> urls.txt
    echo ${BASE_URL}/${run}/per_instance_stats.json   >> urls.txt
    echo ${BASE_URL}/${run}/run_spec.json             >> urls.txt
    echo ${BASE_URL}/${run}/scenario.json             >> urls.txt
    echo ${BASE_URL}/${run}/scenario_state.json       >> urls.txt
    echo ${BASE_URL}/${run}/stats.json                >> urls.txt
done

wget -x -nH -i urls.txt
# aria2c would be faster ...

rm urls.txt