.. _start:

===============
Getting Started
===============

This page shows how to get started with Mariner.


.. _install:

Installation
============

The project runs in a fully containarized environment in such a way
it can run from any major OS.

Using ``docker compose``
------------------------

1. Install Docker v23.0.4 and Docker Compose v2.17.2. Other versions
   of ``docker`` may work but ``docker compose`` should be higher than 2.
2. Build the needed docker images

.. code-block:: console

   docker compose build --parallel backend webapp

3. Start the services. Using ``--build``

.. code-block:: console

   docker compose up --build backend webapp

Using ``make``
--------------

Our Makefile uses docker compose under the hood.

1. Install Docker v23.0.4 and Docker Compose v2.17.2. Other versions
   of ``docker`` may work but ``docker compose`` should be higher than 2.
2. Start

.. code-block:: console

   make start

Using local tools
-----------------

This approach has only been tested on unix based systems.

.. todo::
   Add instructions on how to run locally

.. _envfiles:


Creating Default User
=====================

To create a local user and use the application with basic credentials ``create_admin_user``
can be executed through the CLI. It's important to make sure the database is running, as well the necessary environment
variables are present when the command is executed. Therefore, it's easier to execute the command from docker.

The following snippet can be used to create the admin using docker compose.

.. code-block:: console

	docker compose run --entrypoint python backend -c 'from mariner.db.init_db import create_admin_user; create_admin_user()'

There is also a utility with ``make``

.. code-block:: console

	make create-admin


Configuring Environment
=======================

Environment variables are divided into 2 files, ``backend/.env`` and ``backend/.env.secret``.
The separation was made to support some CI workflows, but all variables should be considered
sensitive in production.

The ``.env`` file contains all variables that are not sensitive, and can be shared with the team.

The ``.env.secret`` file contains all sensitive variables, and should be kept secret.


.. confval:: ENV

   Describes the application environment. One of "production", "development"

.. confval:: SERVER_NAME

   The host used by the backend

.. confval:: SERVER_HOST

   The backend url.

.. confval:: PROJECT_NAME

   The name of the project, used to fill variables in generated files.

.. confval:: DOMAIN

   Same as SERVER_NAME.

   .. todo::

      Choose SERVER_NAME or DOMAIN variables and remove the duplicated.

.. confval:: ALLOWED_GITHUB_AUTH_EMAILS
   :default: ``[]``

   Subsets the emails allowed through github OAuth.

.. confval:: SECRET_KEY

   Used to sign JWT tokens. Should be kept secret from everyone.

.. confval:: APPLICATION_SECRET

   Used as basic auth password for inter service communication.

.. confval:: BACKEND_CORS_ORIGIN

   Defines the origins allowed to use the REST API.

.. confval:: POSTGRES_SERVER

   Defines the host running the postgres server.

.. confval:: POSTGRES_USER

   The username used to connect to the database.

.. confval:: POSTGRES_PASSWORD

   The password used to connect to the database.

.. confval:: POSTGRES_DB

   The database name.

.. confval:: RAY_ADDRESS

   The URI used to connect ray, e.g. ``ray://ray-head:10001``

.. confval:: MLFLOW_TRACKING_URI

   The MLFlow's server tracking URI.

.. confval:: MLFLOW_ARTFIFACT_URI

   The MLlow's artifact URI. Used to store models and experiments metadata.

.. confval:: LIGHTNING_LOGS_DIR

   Can be either a S3 URI or a file path. Used to store the outputs of lightning loggers.

.. confval:: GITHUB_CLIENT_ID

   Configures authentication by Github OAuth.

.. confval:: GITHUB_CLIENT_SECRET

   Configures authentication secret by Github OAuth.

.. confval:: AWS_MODE
   :default: ``"local"``

   Either ``local`` or ``sts``. If ``local``, search credentials from environment variables named ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``. If ``sts`` uses `Security Token Service <https://docs.aws.amazon.com/STS/latest/APIReference/welcome.html>`_ to generate temporary credentials.

.. confval:: AWS_ACCRESS_KEY_ID

   Key id of the AWS credentials.

.. confval:: AWS_SECRET_ACCESS_KEY

   Key secret of the AWS credentials.

.. confval:: AWS_REGION

   AWS region where cloud services operates.

.. confval:: AWS_DATASETS

   The path withing S3 where datasets are stored.

   .. warning::

      Should not include S3 uri schema `s3://`.
      Example that will work: ``dev-mariner-datasets``
      Example that fails: ``s3://dev-mariner-datasets``

   .. todo::

      It will work better as S3 schema.

.. confval:: AWS_MODELS_BUCKET

   S3 URI used to store models.

.. confval:: API_V1_STR
   :default: ``"/api/v1"``

.. confval:: ACCESS_TOKEN_EXPIRE_MINUTES
   :default: ``12888`` equilaent to 8 days

   S3 URI used to store models.

.. confval:: EMAILS_ENABLED

   ???
