#!/bin/sh -e
set -x

echo Running format script
echo $@

isort --check-only $@
autoflake --quiet --check --ignore-init-module-imports --remove-all-unused-imports --recursive --remove-unused-variables --in-place $@
black --check $@
