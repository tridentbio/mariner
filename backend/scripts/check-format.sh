#!/bin/sh -e
set -x
autoflake --quiet --check --ignore-init-module-imports --remove-all-unused-imports --recursive --remove-unused-variables --in-place . 
black --check . $@
isort --check-only . $@
