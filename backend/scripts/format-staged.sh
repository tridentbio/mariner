#!/bin/sh -e
set -x

# Script to format only staged files

# Get the names of the staged files
staged_files=$(git diff --cached --name-only --diff-filter=ACM)

# Run autoflake, black, and isort on the staged files
echo "$staged_files" | xargs autoflake --remove-all-unused-imports --recursive --remove-unused-variables --exclude=__init__.py --in-place
echo "$staged_files" | xargs black
echo "$staged_files" | xargs isort
