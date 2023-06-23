#!/usr/bin/env bash

set -x
flake8
mypy --explicit-package-bases --namespace-packages . 
pylint --extension-pkg-whitelist='pydantic' .
