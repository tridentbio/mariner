#!/bin/sh

sh /app/scripts/prestart.sh && poetry run python -m api.main

