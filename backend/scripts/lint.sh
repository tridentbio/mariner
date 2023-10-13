#!/usr/bin/env bash

set -e
pylint --extension-pkg-whitelist='pydantic' .
