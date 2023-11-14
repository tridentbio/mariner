#!/usr/bin/env bash

set -e

pylint $@
PYLINT_RESULT=$?


if [[ $PYLINT_RESULT > 0 ]]
then
  echo  "pylint failed with status $PYLINT_RESULT"
  exit 1
fi
