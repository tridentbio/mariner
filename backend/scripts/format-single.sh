#!/bin/bash
set -e

isort $@
autoflake --in-place --remove-all-unused-imports --recursive --remove-unused-variables --exclude=__init__.py $@
black $@
