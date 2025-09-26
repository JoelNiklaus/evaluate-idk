#!/bin/bash

# Get the API key from the .env file
export OPENROUTER_API_KEY=$(cat .env | grep OPENROUTER_API_KEY | cut -d '=' -f 2)
export PYTHONPATH=$(pwd):$PYTHONPATH

export EVALUATE_IDK_LOG_LEVEL=DEBUG
export LITELLM_LOG=DEBUG

MODEL_NAMES=(
  #"openai/gpt-4.1-mini"
  "google/gemini-2.5-pro" 
  #"deepseek/deepseek-v3.1-terminus"
  #"moonshotai/kimi-k2-0905"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  lighteval endpoint litellm \
    "provider=openrouter,model_name=openrouter/$MODEL_NAME,concurrent_requests=1,api_key=$OPENROUTER_API_KEY" \
    "community|gpqa-diamond-idk|0" \
    --custom-tasks gpqa_diamond_idk.py \
    --dataset-loading-processes 1 \
    --save-details
done
#  --max-samples 100 \

