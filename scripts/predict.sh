#!/bin/bash

args=(
  --do_predict True
  --num_return_sequences 1
)
bash ./run.sh "${args[@]}" "$@"
