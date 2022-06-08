#!/bin/sh -e
python -m app.features.model.generate base > app/features/model/layers.py
