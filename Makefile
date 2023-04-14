# To see what the make can do in this project
# run make `help`

# Default target to run
.DEFAULT_GOAL:=help

.PHONY: help build create-admin start pre-commit pre-commit-install pre-commit-uninstall migrate-backend migrate-mlflow backend-install webapp-install

# docker compose prefix command (useful to run with
# different environments or docker compose configurations
DOCKER_COMPOSE = docker compose -f docker-compose.yml -f docker-compose.override.yml

# the main services needed to interact with the application
CORE_SERVICES = db mlflowdb mlflow ray-head ray-worker backend webapp

# a variable to control what services are passed as input to some commands
SERVICES = all

# Allows to pass extra arguments to some commands.
ARGS =

# Files to check if it's needed to reinstall dependencies
BACKEND_DEPENDENCY_FILES = backend/pyproject.toml backend/poetry.lock
WEBAPP_DEPENDENCY_FILES = webapp/package.json webapp/package-lock.json

##@ Dependencies

webapp-install:         ## Install dependencies to run webapp locally and run webapp tools
	cd webapp &&\
		npm install


backend-install:        ## Install dependencies to run backend locally and run it's CLI tools
	cd backend &&\
		poetry install &&\
		poetry run install_deps_cpu

help:                   ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

ps:                   ## Gets all services status
	$(DOCKER_COMPOSE) ps

logs:                   ## Watches logs from running services. The variable SERVICE controls the services to attach, and defaults to all running
ifeq ($(SERVICES),all)
	$(DOCKER_COMPOSE) logs --follow $(ARGS)
else
	$(DOCKER_COMPOSE) logs --follow $(SERVICES) $(ARGS)
endif

build:                  ## Builds needed local images to run application.
	$(DOCKER_COMPOSE) build --parallel backend webapp

create-admin:           ## Creates default "admin@mariner.trident.bio" with "123456" password
	$(DOCKER_COMPOSE) run --entrypoint "python -c 'from mariner.db.init_db import create_admin_user; create_admin_user()'" backend

start:                  ## Starts services (without building them explicitly)
	$(DOCKER_COMPOSE) up --wait $(CORE_SERVICES)

stop:                   ## Stops running docker-compose services
	$(DOCKER_COMPOSE) stop

migrate-backend:        ## Runs mariner alembic migrations
	$(DOCKER_COMPOSE) run --entrypoint "alembic upgrade head" backend

migrate-mlflow:         ## Runs mlflow alembic migrations
	$(DOCKER_COMPOSE) run mlflow mlflow db upgrade postgresql://postgres:123456@mlflowdb:5432/app

migrate: migrate-backend migrate-mlflow   ## Runs all migrations

test-backend:           ## Runs all tests in the backend (integration and unit)
	$(DOCKER_COMPOSE) exec backend pytest $(ARGS)

test-backend-unit:           ## Runs backend unit tests
	$(DOCKER_COMPOSE) exec backend pytest -m 'not integration'

test-webapp-unit: webapp-install ## Runs webapp unit tests
	cd webapp && npm run test:unit

test-integration: start          ## Runs unit tests
	$(DOCKER_COMPOSE) exec backend pytest -m 'integration' $(ARGS)

test-e2e: webapp-install start  ## Runs e2e cypress tests
	cd webapp && npx cypress run


pre-commit: backend-install frontend-install ## Runs pre-commit hooks (formatting, linting and unit testing)
	pre-commit

pre-commit-install:     ## Installs pre-commit as a git hook in .git directory
	pip install pre-commit && pre-commit install

pre-commit-uninstall:   ## Removes pre-commit managed git hooks from .git directory
	pre-commit uninstall
