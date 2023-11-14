#!/bin/bash
set -e

isort --diff -c $@
autoflake --check --ignore-init-module-imports --remove-all-unused-imports --recursive --remove-unused-variables --in-place $@
black --diff --check $@
