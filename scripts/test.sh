docker-compose --env-file .env.test -f docker-compose-cicd.yml \
  exec -T backend poetry run coverage run -m pytest
docker-compose --env-file .env.test -f docker-compose-cicd.yml \
  exec -T backend poetry run coverage json
docker-compose --env-file .env.test -f docker-compose-cicd.yml \
  exec -T backend poetry run coverage html
docker-compose --env-file .env.test -f docker-compose-cicd.yml \
  exec -T backend poetry run coverage report && \
  echo 'Test coverage is good!' || \
  (echo 'Test coverage is under limit' && exit 1)
