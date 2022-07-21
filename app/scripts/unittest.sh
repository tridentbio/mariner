#!/usr/bin/env bash

set -e
set -x

pytest --cov=app.features --cov=app.core --cov-report=term-missing "${@}"
