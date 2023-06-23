#!/usr/bin/env bash

# Script to run tests with coverage.
#
# Run test suites
echo coverage run "$@"
coverage run "$@"
TEST_RESULT=$?

# Populate cov/json if JSON_OUT is not empty
[ ! -z $JSON_OUT ] && coverage json -o $JSON_OUT

# Populate cov/html
coverage html
coverage report

if [[ $TEST_RESULT != 0 ]];
then
  echo "Some test failed or coverage is not met!";
  exit 1
fi

