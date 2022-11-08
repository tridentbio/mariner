#!/usr/bin/env bash

set -e
set -x
pytest --cov=mariner --cov-report=html "${@}" app/tests/features app/tests/core
