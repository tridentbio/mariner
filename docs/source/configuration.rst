.. _configuration:

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
