docker-compose -f docker-compose-cicd.yml exec backend poetry run coverage run -m pytest 
docker-compose -f docker-compose-cicd.yml exec backend poetry run coverage json
docker-compose -f docker-compose-cicd.yml exec backend poetry run coverage html
docker-compose -f docker-compose-cicd.yml exec backend poetry run coverage report && echo 'Test coverage is good!' || echo 'Test coverage is under 80'
