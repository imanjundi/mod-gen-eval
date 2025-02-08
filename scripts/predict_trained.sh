#!/bin/bash

args=(
  --load_peft_model True
  --model_name_or_path results/train
)
bash ./predict.sh "${args[@]}" "$@"
