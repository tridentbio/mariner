.. _configuration:

Configuring Environment
=======================

The app configuration is done by the :ref:`mariner.core.config` module. It loads information from the environment variables, as well as the pyproject.toml file.

Environment variables are divided into 2 files, ``backend/.env`` and ``backend/.env.secret``.
The separation was made to support some CI workflows, but all variables should be considered
sensitive in production.
The ``.env`` file contains all variables that can be shared with the team.
The ``.env.secret`` file contains all sensitive variables, and should be kept secret.

Server settings
---------------

This configuration is loaded from the environment variables described next into the class `mariner.core.config`_.


.. confval:: SERVER_HOST

   The backend url, such as ``"https://dev.mariner.backend.com"``

.. confval:: BACKEND_CORS_ORIGIN

   A comma separated string with the origins allowed to use the REST API, such as ``"http://localhost:3000,http://localhost:8080"``

.. confval:: ACCESS_TOKEN_EXPIRE_MINUTES


   Duration of token's validity.

   :default: ``12888`` equivalent to 8 days

.. confval:: ENV

   Describes the application environment. One of "production", "development"


Webapp Settings
---------------

Webapp related information the backend needs.

.. confval:: WEBAPP_URL

   The URL used by the webapp. Necessary when the backend needs to redirect to the webapp, such as during oauth flows. Example: ``https://dev.mariner.webapp.com``

   :default: ``http://localhost:3000``

Tenant Settings
---------------

Settings related to the tenant that is currently being used.

.. confval:: TENANT_NAME
   :default: ``default``
   The name of the tenant. Any string will be accepted.

Secret
------

.. confval:: SECRET_KEY

   Used to sign JWT tokens. Should be kept secret and be cryptographic safe.

.. confval:: APPLICATION_SECRET

   Used as basic auth password for inter service communication. Should be kept secret and be cryptographic safe.

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

Services
--------

.. confval:: POSTGRES_URI

   The URI used to connect to the postgres database. Example: ``postgresql://user:password@localhost:5432/dbname``

.. confval:: MLFLOW_POSTGRES_URI

    The URI used to connect to the MLFlow's postgres database. Example: ``postgresql://user:password@localhost:5432/dbname``

   .. warning::
  
    Maybe this is not used. Instead we pass the value of the mlflow postgres directly in the mlflow start command.

.. confval:: MLFLOW_ARTIFACT_URI

    The URI used to connect to the MLFlow's artifact database. Example: ``postgresql://user:password@localhost:5432/dbname``

   .. warning::
  
    Maybe this is not used. Instead we pass the value of the artifact uri directly in the mlflow start command.

.. confval:: MLFLOW_TRACKING_URI

    The URI used to connect to the MLFlow's tracking database. See <https://mlflow.org/docs/latest/tracking.html#id31> for more information.

    

.. confval:: RAY_ADDRESS

    The URI used to connect to the Ray cluster.

Package settings
----------------

This configuration comes from the `backend/pyproject.toml` file, and is loaded by the `mariner.core.config.Package`_ class.


.. confval:: LIGHTNING_LOGS_DIR

   Can be either a S3 URI or a file path. Used to store the outputs of lightning loggers.

.. confval:: API_V1_STR
   :default: ``"/api/v1"``

