#! /usr/bin/env bash
# Only ran from container

# Let the DB start
python /app/api/backend_pre_start.py

# Run migrations
alembic upgrade head

# Create initial data in DB
python /app/mariner/initial_data.py
