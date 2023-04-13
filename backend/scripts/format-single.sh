#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --exclude=__init__.py $@
black $@
isort $@
