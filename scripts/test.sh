#!/usr/bin/env bash
# Script to run tests with coverage.
#
# Run test suites
echo coverage run "$@"
coverage run "$@"
TEST_RESULT=$?
# Populate cov/json
coverage json
# Populate cov/html
coverage html
coverage report
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

