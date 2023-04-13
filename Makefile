# A makefile with commands using docker-compose
DOCKER_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.override.yml

.PHONY: help build create-admin start pre-commit pre-commit-install pre-commit-uninstall

# Shows this help message
help:           ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

# Builds application to run project locally
build:
	$(DOCKER_COMPOSE) build --parallel backend webapp

# Creates account with admin@mariner.trident.bio email and 123456 password for tests and local development purposes
create-admin:
	$(DOCKER_COMPOSE) run --entrypoint "python -c 'from mariner.db.init_db import create_admin_user; create_admin_user()'" backend

# Starts services and wait for all to be healthy
start:
	$(DOCKER_COMPOSE) up --wait db mlflowdb mlflow ray-head ray backend webapp

# TODO
pre-commit:
	cd library1 && pre-commit run --files ./*
	cd library2 && pre-commit run --files ./*

# TODO
pre-commit-install:
	cd library1 && pre-commit install --allow-missing-config
	cd library2 && pre-commit install --allow-missing-config

# TODO
pre-commit-uninstall:
	cd library1 && pre-commit uninstall
	cd library2 && pre-commit uninstall
