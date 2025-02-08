#!/bin/bash

args=(
  --do_eval True
#  --eval_max_new_tokens 16
)
bash ./eval_original.sh "${args[@]}" "$@"
