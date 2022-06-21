#!/bin/sh -e
python -m app.features.model.generate base > app/features/model/schema/layers_schema.py
