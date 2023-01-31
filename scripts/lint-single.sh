#!/usr/bin/env bash

flake8 $@
FLAKE_RESULT=$?
mypy --explicit-package-bases --namespace-packages $@
MYPY_RESULT=$?
pylint --extension-pkg-whitelist='pydantic' $@
PYLINT_RESULT=$?

if [[ $FLAKE_RESULT != 0 ]];
then
  echo "Flake failed with status $FLAKE_RESULT"
fi
if [[ $MYPY_RESULT != 0 ]]
then 
  echo "mypy failed with status $MYPY_RESULT"
fi

if [[ $PYLINT_RESULT > 0 ]]
then
  echo  "pylint failed with status $PYLINT_RESULT"
fi


if [[ $FLAKE_RESULT != 0 || $MYPY_RESULT != 0 || $PYLINT_RESULT > 0 ]];
then
  exit 1
fi
