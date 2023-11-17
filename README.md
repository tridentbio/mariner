# Mariner

Mariner is an application to create and manage ML models from a web interface, without the requirement of knowing how to code to build and use these models.
It works by abstracting the ML tasks, such as training and validating a model, into a simpler functions and json convertible objects, that are used to build a REST API to perform such tasks.


# Getting started

## Installation

To run the application, you simply need:

- [Docker](https://www.docker.com/).
- [Docker Compose](https://docs.docker.com/compose/install/).
- and optionally [GNU Make](https://www.gnu.org/software/make/), which is included in many distributions, and will allow you to use shorter commands

## Starting the application locally

First you should place the AWS credentials in the `backend/.env.secret` file so the docker-compose files can use them in the services that must interact with AWS.
For production environments, those credentials should be in the environment variables. The necessary roles

Use one of the following to start all core services:

```console
make start
```

or 

```console
docker compose up --wait backend webapp` is also another way that does not require building.
```

Alternatively you may wish to run only the backend on docker, and the webapp locally. Then you can:

```console
make start-backend
cd webapp
npm install . 
npm start
```

In case you're not using make you can omit the webapp service from the `docker-compose` start command.

Finally, you'll want a local user credentials to interact with the app. For that you can run one of the following commands:

```
make create-admin
```

or 

```
docker compose run --entrypoint "python -c 'from mariner.db.init_db import create_admin_user; create_admin_user()'" backend
```

This will create a user with email `admin@.mariner.trident.bio` and password `123456`.

Finally, access <http://localhost:3000/login> and login to use the app. Checkout the User Guide to know what
