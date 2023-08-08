#!/usr/bin/env bash

echo Linting $@
pylint --extension-pkg-whitelist='pydantic' $@
PYLINT_RESULT=$?


if [[ $PYLINT_RESULT > 0 ]]
then
  echo  "pylint failed with status $PYLINT_RESULT"
  exit 1
fi
