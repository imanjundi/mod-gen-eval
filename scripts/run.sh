#!/bin/bash

args=(
#  --model_type mistralai/Mistral-7B-Instruct-v0.1
#  --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1
#  -- use_system_role False
#  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct
  --model_name_or_path meta-llama/Meta-Llama-3-70B-Instruct
#  --moderation_type quality broadening
  --moderation_type moderation
#  --output_dir results/no-repeat
#  --overwrite_output_dir True
  )
python baseline.py "${args[@]}" "$@"