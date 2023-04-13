#!/bin/sh

poetry run mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root $MLFLOW_ARTIFACT_URI --backend-store-uri $MLFLOW_DB_DIALECT://$MLFLOW_DB_USERNAME:$MLFLOW_DB_PASSWORD@$MLFLOW_DB_HOST:$MLFLOW_DB_PORT/$MLFLOW_DB_DATABASE