# Mariner - Backend

Here is some information about deploying using this chart

### Healthcheck URL

`/health`

You need to create two configmaps with the following variables

### ConfigMap  1

the name of this configmap needs to be the same as the implementation of this helm chart, let's use the name mariner-backend as an example

| Environment Variabel | Description | Value |
| --- | --- | --- |
| AWS_DATASETS | Bucket name to store the datasets | dev-mariner-datasets |
| AWS_MODELS_BUCKET | Bucket name to store the datasets | s3://dev-mariner-datasets |
| AWS_REGION | Region | us-east-1 |
| BACKEND_CORS_ORIGINS | Domain names to allow CORS | '["https://dev.mariner.trident.bio/", "https://backend.dev.mariner.trident.bio/"]' |
| DOMAIN | Domain name | backend.dev.mariner.trident.bio |
| EMAILS_FROM_EMAIL | Email to send notification | info@mariner.trident.bio |
| ENV | Environment | DEV |
| FIRST_SUPERUSER | First super user of mariner | admin@mariner.trident.bio |
| GITHUB_CLIENT_ID | Gitub OAuth App ID |  |
| ID | Random ID for the app | 0l551989a1e373k7f1343a |
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
| WEBAPP_URL | Frontend URL | https://dev.mariner.trident.bio/ |

### ConfigMap  2

the name of this configmap needs to be the same as the implementation of this helm chart, adding “-emails” at the end

| Environment Variabel | Description | Value |
| --- | --- | --- |
| ALLOWED_GITHUB_AUTH_EMAILS | List of allowed emails for frontend login with Github SSO | ["tshimko@trident.bio"] |