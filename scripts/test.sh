#!/usr/bin/env bash
# Script to run tests with coverage.
#
# Run test suites
poetry run coverage run $@
TEST_RESULT=$?
# Populate cov/json
poetry run coverage json
# Populate cov/html
poetry run coverage html
poetry run coverage report
COVERAGE_RESULT=$?

if [[ $TEST_RESULT != 0 ]];
then
  echo "Some test failed!";
  exit 1
fi

if [[ $COVERAGE_RESULT != 0 ]];
then
  echo "Test coverage is not met";
  exit 1
fi

