#!/usr/bin/env bash

set -x
pylint --extension-pkg-whitelist='pydantic' .
