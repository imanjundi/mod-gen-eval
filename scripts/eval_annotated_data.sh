#!/bin/bash

args=(
  --dataset aggregated_annotated
#  does not work with that well
#  --use_annotation_form_for_eval True
)
bash ./eval_original.sh "${args[@]}" "$@"
