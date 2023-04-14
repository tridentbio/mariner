# A makefile with commands using docker-compose

.DEFAULT_GOAL:=help

DOCKER_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.override.yml

CORE_SERVICES = db mlflowdb mlflow ray-head ray backend webapp

.PHONY: help build create-admin start pre-commit pre-commit-install pre-commit-uninstall


##@ Dependencies

backend-tools: backend/pyproject.toml backend/poetry.lock
	cd backend &&\
		poetry install --only dev


help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

logs:  ## Watches logs from running services.
	$(DOCKER_COMPOSE) logs --follow

build: ## Builds needed local images to run application.
	$(DOCKER_COMPOSE) build --parallel backend webapp

create-admin: ## Creates default "admin@mariner.trident.bio" with "123456" password
	$(DOCKER_COMPOSE) run --entrypoint "python -c 'from mariner.db.init_db import create_admin_user; create_admin_user()'" backend

start: ## Starts services (without building them explicilty)
	$(DOCKER_COMPOSE) up --wait $(CORE_SERVICES)

stop:
	$(DOCKER_COMPOSE) stop

migrate-backend:
	$(DOCKER_COMPOSE)

# TODO
pre-commit: backend-tools
	pre-commit

# TODO
pre-commit-install: backend-tools
		pre-commit install --allow-missing-confikg

# TODO
pre-commit-uninstall:
	cd library1 && pre-commit uninstall
	cd library2 && pre-commit uninstall
