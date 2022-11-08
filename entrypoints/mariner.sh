#!/bin/sh

poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 80


