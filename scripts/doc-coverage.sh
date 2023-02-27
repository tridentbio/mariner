#!/usr/bin/bash

[ -f $BASE_COV_PATH ] && BASE_COV=$(cat $BASE_COV_PATH) || BASE_COV=0
COV=$(docstr-coverage mariner/ model_builder/ api/ --skip-init --skip-magic --skip-private -p)
if (( $(echo "$COV < $BASE_COV" | bc) )); then
  exit 1
else
  echo $COV
fi

