#!/usr/bin/env bash

pylint --extension-pkg-whitelist='pydantic' $@
PYLINT_RESULT=$?

if [[ $PYLINT_RESULT > 0 ]]
then
  echo  "pylint failed with status $PYLINT_RESULT"
fi


if [[ $PYLINT_RESULT > 0 ]];
then
  exit 1
fi
