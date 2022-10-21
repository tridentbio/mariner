#!/usr/bin/env bash
set -e
set -x
pytest --cov-branch --cov=app.api --cov=mariner --cov=model_builder --cov-report=html "${@}" app/tests/api
