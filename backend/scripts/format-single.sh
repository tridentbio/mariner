#!/bin/sh -e
set -x

isort $@
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --exclude=__init__.py $@
black $@
