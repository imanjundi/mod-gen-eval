#!/bin/bash

args=(
  --do_eval_original False
  --do_eval True
  --eval_generated_input_dir /your/path/to/output_dir
)
bash ./eval_annotated_data.sh "${args[@]}" "$@"
