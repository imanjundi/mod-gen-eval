#!/bin/bash

args=(
  --dataset annotated
  --use_gen_instructions_for_eval False
  --use_annotation_instructions True
  --use_annotation_intro_answers_for_eval True
#  does not work with that well
#  --use_annotation_form_for_eval True
)
bash ./eval_original.sh "${args[@]}" "$@"
