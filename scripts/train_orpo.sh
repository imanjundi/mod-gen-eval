#!/bin/bash

args=(
  --training_method orpo
)
bash ./train.sh "${args[@]}" "$@"