.. _configuration:

=======================
Configuring Environment
=======================

The app configuration is done by the :doc:`mariner.core.config </generated/mariner.core.config>` module. It loads information from the environment variables, as well as the pyproject.toml file.

Environment variables are divided into 2 files, ``backend/.env`` and ``backend/.env.secret``.
The separation was made to support some CI workflows, but all variables should be considered
sensitive in production.
The ``.env`` file contains all variables that can be shared with the team.
The ``.env.secret`` file contains all sensitive variables, and should be kept secret.

In the implementation, those variables are usually accessed with ``get_app_settings``

General Configuration
---------------------

.. confval:: SERVER_HOST

   The backend url, such as ``"https://dev.mariner.backend.com"``

.. confval:: SERVER_CORS 

   A comma separated string with the origins allowed to use the REST API, such as ``"http://localhost:3000,http://localhost:8080"``

.. confval:: ACCESS_TOKEN_EXPIRE_MINUTES


   Duration of token's validity.

   :default: ``12888`` equivalent to 8 days


.. confval:: WEBAPP_URL

   The URL used by the webapp. Necessary when the backend needs to redirect to the webapp, such as during oauth flows. Example: ``"https://dev.mariner.webapp.com"``

   :default: ``"http://localhost:3000"``

.. confval:: TENANT_NAME

   The name of the tenant. Any string will be accepted.

   :default: ``"default"``

.. confval:: LIGHTNING_LOGS_DIR

   Environment variable used to specify where the lightning trainer should store the generated files (used in the ``default_root_dir`` parameter of the Trainer). Can be a s3 uri, such as s3://dev-mariner-datasets/lightning-logs

  :default: ``"lightning_logs/"``, i.e. stores the files in the local filesystem. Not recommended for production environments.

Services
--------

.. confval:: POSTGRES_URI

   The URI used to connect to the postgres database. Example: ``postgresql://user:password@localhost:5432/dbname``

.. confval:: MLFLOW_TRACKING_URI

    The URI used to connect to the MLFlow's tracking database. See `the mlflow docs <https://mlflow.org/docs/latest/tracking.html#id31>`_ for more information.
    Example: ``"http://localhost:5000"`` when running mlflow locally.

    
.. confval:: RAY_ADDRESS

    The URI used to connect to the Ray cluster. Example: ``"ray://ray-head-backend.ray.svc.cluster.local:10001"``

OAuth Settings
--------------

Here we describe the environment variables that have a role in the OAuth flow.
New OAuth providers can be added by adding the variables to the environment
and providing an implementation for the authentication flow in the oauth_providers module.
All OAuth providers must have the following variables.

- ``OAUTH_<PROVIDER-ID>_NAME``: Configures the name of the OAuth provider button in the frontend.
- ``OAUTH_<PROVIDER-ID>_CLIENT_ID``: Used to identify the application in the OAuth provider.
- ``OAUTH_<PROVIDER-ID>_CLIENT_SECRET``: Used to authenticate the application in the OAuth provider.
- ``OAUTH_<PROVIDER-ID>_AUTHORIZATION_URL``: The URL used to start the OAuth flow.
- ``OAUTH_<PROVIDER-ID>_SCOPE``: The scope of the OAuth flow.
- ``OAUTH_<PROVIDER-ID>_ALLOWED_EMAILS``: Optional list of emails that are allowed separated by strings. Example: ``"user1@domain.com,user2@domain.com"``

Those configurations are used in the :doc:`/generated/oauth_providers` module to configure the OAuth flow.

Secrets
-------

All following variables are considered sensitive and should be kept secret.

.. confval:: AUTHENTICATION_SECRET_KEY

   Used to sign JWT tokens. Should be kept secret and be cryptographic safe.

.. confval:: DEPLOYMENT_URL_SIGNATURE_SECRET_KEY

   Used to sign deployment urls. Should be kept secret and be cryptographic safe.

.. confval:: APPLICATION_SECRET

   Used as basic auth password for inter service communication. Should be kept secret and be cryptographic safe.

AWS
---

For the application to work, the AWS credentials must have permission to read and write to the S3 buckets defined in ``AWS_DATASETS`` and ``AWS_MODELS_BUCKET``.

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



Production Environments
=======================

For production environments, we have the `infrastructure/` folder distributed with the project. It may have information from past deployments, so it should be reviewed before using it.
To deploy it on AWS, the necessary services are:

- S3 buckets of ``AWS_DATASETS`` and ``AWS_MODELS_BUCKET``. Could be the same bucket.
- RDS
- EKS
- Route53
- Cloudwatch
- ECR

With read and write permissions to these service, the infrastructure can be deployed using helm and kubectl.
