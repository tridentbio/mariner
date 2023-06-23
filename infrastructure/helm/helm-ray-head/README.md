# Ray Head

Here is some information about deploying using this chart

### ConfigMap  Ray Head

You need to create two configmaps with the following variables

the name of this configmap needs to be the same as the implementation of this helm chart, let's use the name `ray-head` as an example

| Environment Variabel | Description | Value |
| --- | --- | --- |
| AWS_DATASETS | Bucket name to store the datasets | dev-mariner-datasets |
| AWS_REGION | Region | us-east-1 |
| AWS_MODELS_BUCKET | Bucket name to store the datasets | s3://dev-mariner-datasets |
| BACKEND_CORS_ORIGINS | Domain names to allow CORS | '["https://dev.mariner.trident.bio/", "https://backend.dev.mariner.trident.bio/"]' |
| DOMAIN | Domain name | backend.dev.mariner.trident.bio |
| EMAILS_FROM_EMAIL | Email to send notification | info@mariner.trident.bio |
| FLOWER_BASIC_AUTH: | Basic Auth Flower | admin:123456 |
| FIRST_SUPERUSER | First super user of mariner | admin@mariner.trident.bio |
| ID | Random ID for the app | 0805f51ee373c7f1343a |
| MLFLOW_ARTIFACT_URI | S3 Buckets with datasets | s3://dev-mariner-datasets |
| MLFLOW_TRACKING_URI | URL of MLFLOW inside or outside cluster | http://mlflow-default.default.svc.cluster.local:5000/ |
| PROJECT_NAME | Name of Project | Mariner |
| RAY_ADDRESS | Service name of Ray Head inside cluster | ray-head-backend |
| RAY_HEAD_PORT | Port of Ray head | 6379 |
| RAY_NUM_CPU | Number of CPU | 1 |
| SENTRY_DSN | DSN of Sentry | "" |
| SERVER_HOST | URL where the app will be exposed | https://backend.dev.mariner.trident.bio/ |
| SERVER_NAME | DNS where the app will be exposed | backend.dev.mariner.trident.bio |
| SMTP_HOST | Host SMTP | http://smtp.gmail.com/ |
| SMTP_PORT | Port of SMTP | "587" |
| SMTP_TLS | Boolean value if TLS is enabled | True |
| SMTP_USER | User SMTP | admin@mariner.trident.bio |
| USERS_OPEN_REGISTRATION |  | False |
| STACK_NAME | Name of stack | mariner-trident-bio |

### ConfigMap  2

the name of this configmap needs to be the same as the implementation of this helm chart, adding “-secret” at the end e.g `ray-head-secrets`

| Environment Variabel | Description | Example |
| --- | --- | --- |
| APPLICATION_SECRET | Random secret for the app | "” |
| AWS_SECRET_KEY | Secret Access Key AWS | "” |
| AWS_SECRET_KEY_ID | Access ID AWS | "” |
| FIRST_SUPERUSER_PASSWORD | Random password | "” |
| POSTGRES_DB | Name of Database | "” |
| POSTGRES_PASSWORD | Password of postgres user | "” |
| POSTGRES_SERVER | Hostname of database | "” |
| POSTGRES_USER: | Username of postgres | "” |
| SECRET_KEY | Random secret for the app | "” |
| SMTP_PASSWORD: | SMTP Password | "” |
| Secret: | Random secret for the app | "” |