#!/bin/bash

args=(
  # only possible data for training for now
  --dataset regroom
  --do_train True
  --do_predict False
  --do_eval False
  --do_eval_original False
)
bash ./run.sh "${args[@]}" "$@"
