#!/bin/sh -e
set -e

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --exclude=__init__.py --in-place .
black .
isort .

