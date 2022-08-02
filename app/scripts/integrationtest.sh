#!/usr/bin/env bash

set -e
set -x
pytest --cov-branch --cov=app.api --cov=app.features --cov=app.core --cov-report=html "${@}" app/tests/api
