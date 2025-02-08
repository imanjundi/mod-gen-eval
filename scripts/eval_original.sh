#!/bin/bash

args=(
  --do_predict False
  --do_eval False
  --do_eval_original True
)
bash ./run.sh "${args[@]}" "$@"
