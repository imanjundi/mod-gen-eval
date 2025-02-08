#!/bin/bash

args=(
  --dataset aggregated_annotated
  --do_eval_original False
)
bash ./predict.sh "${args[@]}" "$@"
