#!/usr/bin/bash

# Script to operate on docker-compose when upgraded with docker-compose-cicd.yml file
# Use this script like the following:
#
#   source scripts/ci-test-helper.sh
#
# Requirements to run this script is docker-compose and a
# .env.secret populated with secret variables
#
#
# In actual Github CI, the secret variables are populated by github secrets


DOCKER_COMPOSE="docker-compose -f docker-compose.yml -f docker-compose-cicd.yml --env-file ./.env.secret"
echo "Compose commands ran with: $DOCKER_COMPOSE"

down() {
  bash -c "$DOCKER_COMPOSE down"
}

setup_cypress() {
  echo "Setting up application for e2e testing..."
  bash -c "$DOCKER_COMPOSE up -d backend"


  echo "Creating admin@mariner.trident.bio user"
  DEFAULT_USER_CREATION_OUT=$(bash -c "$DOCKER_COMPOSE exec -T backend python -c 'from mariner.db.init_db import create_admin_user; create_admin_user()'")
  SUB="DETAIL:  Key (email)=(admin@mariner.trident.bio) already exists."
  
  if grep -q "$SUB" <<< "$DEFAULT_USER_CREATION_OUT"; then
    echo "Admin user already exists"
  else
    echo $DEFAULT_USER_CREATION_OUT
    exit $?
  fi
}

integration_test() {
  bash -c "$DOCKER_COMPOSE up -d backend"
  COMMAND="$DOCKER_COMPOSE exec backend pytest -m 'integration' $@"
  bash -c "$COMMAND"
}

cypress_up() {
  setup_cypress && \
  bash -c "$DOCKER_COMPOSE run cypress --browser chrome $@"
}

echo "Commands: 
  down:             Stops and removes all running containers from this docker-compose file combo 
  setup_cypress:    Runs commands to setup cypress
  cypress_up:       Runs e2e tests on cypress
  integration_test: Runs backend integration tests"
