# MLFlow

Here is some information about deploying using this chart

### Healthcheck URL

`/ping`

### ConfigMap  MLFLow

You need to create two configmaps with the following variables

the name of this configmap needs to be the same as the implementation of this helm chart, let's use the name `mlflow` as an example

| Environment Variabel | Description | Value |
| --- | --- | --- |
| AWS_DATASETS | Bucket name to store the datasets | dev-mariner-datasets |
| AWS_REGION | Region | us-east-1 |
| BACKEND_CORS_ORIGINS | Domain names to allow CORS | '["https://dev.mariner.trident.bio/", "https://backend.dev.mariner.trident.bio/"]' |
| DOMAIN | Domain name | backend.dev.mariner.trident.bio |
| EMAILS_FROM_EMAIL | Email to send notification | info@mariner.trident.bio |
| FIRST_SUPERUSER | First super user of mariner | admin@mariner.trident.bio |
| ID | Random ID for the app | 0805f51ee373c7f1343a |
| MLFLOW_ARTIFACT_URI | S3 Buckets with datasets | s3://dev-mariner-datasets |
| MLFLOW_TRACKING_URI | URL of MLFLOW inside or outside cluster | http://mlflow-default.default.svc.cluster.local:5000/ |
| PROJECT_NAME | Name of Project | Mariner |
| RAY_ADDRESS | URL of Ray Head inside or outside cluster | ray://ray-cluster.ray.svc.cluster.local:10001 |
| SENTRY_DSN | DSN of Sentry | "" |
| SERVER_HOST | URL where the app will be exposed | https://backend.dev.mariner.trident.bio/ |
| SERVER_NAME | DNS where the app will be exposed | backend.dev.mariner.trident.bio |
| SMTP_HOST | Host SMTP | http://smtp.gmail.com/ |
| SMTP_PORT | Port of SMTP | "587" |
| SMTP_TLS | Boolean value if TLS is enabled | True |
| SMTP_USER | User SMTP | admin@mariner.trident.bio |
| USERS_OPEN_REGISTRATION |  | False |
| OAUTH2_PROXY_CLIENT_ID | OAuth app github to MLFLOW | "” |
| OAUTH2_PROXY_HTTP_ADDRESS | Oauth proxy address inside pod | http://0.0.0.0:4180/ |
| OAUTH2_PROXY_PROVIDER | Provider Oauth | github |
| OAUTH2_PROXY_UPSTREAMS | Oauth proxy upstream inside pod | http://127.0.0.1:5000/ |

### ConfigMap  2

the name of this configmap needs to be the same as the implementation of this helm chart, adding “-secret” at the end e.g `mlflow-secrets`

| Environment Variables | Description | Example |
| --- | --- | --- |
| APPLICATION_SECRET | Random secret for the app | "” |
| AWS_ACCESS_KEY_ID | Access ID AWS | "” |
| AWS_SECRET_ACCESS_KEY | Secret Access Key AWS | "” |
| AWS_SECRET_KEY | Secret Access Key AWS | "” |
| AWS_SECRET_KEY_ID | Access ID AWS | "” |
| FIRST_SUPERUSER_PASSWORD | Random password | "” |
| GITHUB_CLIENT_SECRET | Github client secret | "” |
| POSTGRES_DB | Name of Database | "” |
| POSTGRES_PASSWORD | Password of postgres user | "” |
| POSTGRES_SERVER | Hostname of database | "” |
| POSTGRES_USER | Username of postgres | "” |
| SECRET_KEY | Random secret for the app | "” |
| SMTP_PASSWORD | SMTP Password | "” |
| Secret | Random secret for the app | "” |
| MLFLOW_DB_DATABASE | Name of Database | "” |
| MLFLOW_DB_DIALECT | Type of database | "postgresql” |
| MLFLOW_DB_HOST | Hostname of database | "” |
| MLFLOW_DB_PASSWORD | Password of postgres user | "” |
| MLFLOW_DB_PORT | Port of database | "5432” |
| MLFLOW_DB_USERNAME | Username of postgres | "” |
| MLFLOW_TRACKING_TOKEN | Token for MLFlow | "” |
| OAUTH2_PROXY_CLIENT_SECRET | Secret of Oauth app  | "” |
| OAUTH2_PROXY_COOKIE_SECRET | Random cookie secret fo OAuth app | "” |