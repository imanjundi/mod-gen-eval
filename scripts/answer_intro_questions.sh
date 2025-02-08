#!/bin/bash

args=(
  --do_eval False
  --do_eval_original False
  --answer_intro_questions True
)
bash ./eval_original.sh "${args[@]}" "$@"
