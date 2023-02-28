#!/usr/bin/bash

if [ -n $BASE_COV_PATH ] && [ -f $BASE_COV_PATH ]; then
  BASE_COV=$(cat $BASE_COV_PATH) 
else
  BASE_COV=0
fi
COV=$(docstr-coverage mariner/ model_builder/ api/ --skip-init --skip-magic --skip-private -p)
if (( $(echo "$COV < $BASE_COV" | bc) )); then
  echo "New docstring coverage $COV < base docstring doverage $BASE_COV"
  exit 1
else
  echo $COV
fi

