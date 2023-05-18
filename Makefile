# To see what the make can do in this project
# run make `help`

# Default target to run
.DEFAULT_GOAL:=help

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

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -a
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

##@ Dependencies


.PHONY: webapp-install
webapp-install:         ## Install dependencies to run webapp locally and run webapp tools
	cd webapp &&\
		npm ci


.PHONY: backend-install
backend-install:        ## Install dependencies to run backend locally and run it's CLI tools
	cd backend &&\
		poetry install &&\
		poetry run install_deps_cpu

.PHONY: help
help:                   ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)


.PHONY: ps
ps:                   ## Gets all services status
	$(DOCKER_COMPOSE) ps
	

.PHONY: logs
logs:                   ## Watches logs from running services. The variable SERVICE controls the services to attach, and defaults to all running
ifeq ($(SERVICES),all)
	$(DOCKER_COMPOSE) logs --follow $(ARGS)
else
	$(DOCKER_COMPOSE) logs --follow $(SERVICES) $(ARGS)
endif


.PHONY: build
build:                  ## Builds needed local images to run application.
	$(DOCKER_COMPOSE) build --parallel backend webapp
	$(DOCKER_COMPOSE) build --parallel ray-head ray-worker mlflow mlflowdb db


.PHONY: create-admin
create-admin:           ## Creates default "admin@mariner.trident.bio" with "123456" password
	$(DOCKER_COMPOSE) run --entrypoint "python -c 'from mariner.db.init_db import create_admin_user; create_admin_user()'" backend

start-backend:         ## Builds and starts backend
	$(DOCKER_COMPOSE) build backend
	$(DOCKER_COMPOSE) up --wait backend

.PHONY: start
start:                  ## Starts services (without building them explicitly)
	$(DOCKER_COMPOSE) up --wait $(CORE_SERVICES) 


.PHONY: stop
stop:                   ## Stops running docker-compose services
	$(DOCKER_COMPOSE) stop


.PHONY: migrate-backend
migrate-backend:        ## Runs mariner alembic migrations
	$(DOCKER_COMPOSE) run --entrypoint "alembic upgrade head" backend


.PHONY: migrate-mlflow
migrate-mlflow:         ## Runs mlflow alembic migrations
	$(DOCKER_COMPOSE) run mlflow mlflow db upgrade postgresql://postgres:123456@mlflowdb:5432/app


.PHONY: migrate
migrate: migrate-backend migrate-mlflow   ## Runs all migrations

.PHONY: test-backend
test-backend: build start-backend          ## Runs all tests in the backend (integration and unit)
	$(DOCKER_COMPOSE) exec backend pytest $(ARGS)


.PHONY: test-backend-unit
test-backend-unit: start-backend          ## Runs backend unit tests
	$(DOCKER_COMPOSE) exec backend pytest -m 'not integration' $(ARGS)


.PHONY: test-webapp-unit
test-webapp-unit: webapp-install ## Runs webapp unit tests
	cd webapp && npm run test:unit


.PHONY: test-integration
test-integration: start-backend ## Runs unit tests
	$(DOCKER_COMPOSE) exec backend pytest -m 'integration' $(ARGS)


.PHONY: test-e2e
test-e2e: build start create-admin ## Runs test target
	$(DOCKER_COMPOSE) up e2e


.PHONY: test-target
test-target:  ## Runs test target
	echo hello


.PHONY: pre-commit
pre-commit: backend-install webapp-install  ## Runs pre-commit hooks (formatting, linting and unit testing)
	pre-commit


.PHONY: pre-commit-install
pre-commit-install:     ## Installs pre-commit as a git hook in .git directory
	pip install pre-commit && pre-commit install


.PHONY: pre-commit-uninstall
pre-commit-uninstall:   ## Removes pre-commit managed git hooks from .git directory
	pre-commit uninstall


.PHONY: pre-commit-fix
fix:
	pre-commit run --hook-stage manual

.PHONY: jupyter
jupyter: ## Parse RELEASE.md file into mariner events that will show up as notifications
	cd backend &&\
		cat RELEASES.md | $(DOCKER_COMPOSE) run --entrypoint 'python -m mariner.changelog publish' backend

.PHONY: publish
publish: ## Parse RELEASE.md file into mariner events that will show up as notifications
	cd backend &&\
		cat RELEASES.md | $(DOCKER_COMPOSE) run --entrypoint 'python -m mariner.changelog publish' backend


.PHONY: build-docs 
build-docs: Makefile   ## Builds the documentation
	sphinx-apidoc -o source backend/mariner
	sphinx-apidoc -o source backend/fleet
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

