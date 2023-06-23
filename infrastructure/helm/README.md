# Helm - Documentation

# Documentation for Mariner helm charts

This repository contains an example Helm chart for deploying a web application on a Kubernetes cluster.

## Prerequisites

Before using this repository, you must have the following prerequisites:

- A configured and running Kubernetes cluster
- Helm 3 installed on your local machine

## Project structure

The project is organized into various directories and files. Here's an overview of the project structure:

- **templates**: This directory contains the Kubernetes YAML templates for the resources that will be deployed.
- **values.yaml**: This file contains the default values for the chart.

## Configuration

Before deploying the chart, you will need to configure some values in the values.yaml file. Here's an overview of the required configurations:

- **`image`**: The name of the Docker image that will be deployed.
- **`replicas`**: The number of application replicas that will be deployed.
- **`port`**: The port that the application will listen on.
- **`ingress`**: The configuration for the Kubernetes ingress resource.

To configure these values, refer to the  **`values.yaml`** file..

## Deployment

After configuring the values, you can deploy the chart using Helm. Here's an example of how to deploy the chart:

```
helm install my-chart ./helm-chart
```

This will install the chart with the name **`my-chart`** using the default values.

To override the default values, you can use the following command:

```
helm install my-chart ./helm-chart --set key=value
```

Replace **`key`** and **`value`** with the values you want to override.

## Default values

Here are the default values for the **`values.yaml`** file:

```jsx
replicaCount: 3

command: /app/entrypoints/mariner.sh
normal_deploy:
  enabled: true

version: mariner-v1

port: 80

image:
  repository: mariner-dev
  pullPolicy: Always
  tag: latest
  containerPort: 80
  livenessProbe:
    path: /
    port: 80
    periodSeconds: 30
    successThreshold: 1
    failureThreshold: 3
    timeoutSeconds: 1
  readinessProbe:
    path: /
    port: 80
    periodSeconds: 30
    successThreshold: 1
    failureThreshold: 3
    timeoutSeconds: 1

strategy:
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: false
  annotations: {}
  name: ""

podAnnotations:
  product: mariner

podSecurityContext: {}

securityContext: {}
  # capabilities:
  #   drop:
  #   - ALL
  # readOnlyRootFilesystem: true
  # runAsNonRoot: true
  # runAsUser: 1000

service:
  type: ClusterIP
  port: 80
  annotations:
    alb.ingress.kubernetes.io/target-type: ip

ingress:
  enabled: false
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/load-balancer-name: mariner-backend-alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '60'
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: '5'
    alb.ingress.kubernetes.io/success-codes: '200'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '2'
  hosts:
  - host: dev.mariner.trident.bio
  tls:
  - hosts:
    - dev.mariner.trident.bio

istio_url: dev.mariner.trident.bio
istio_enabled: true

resources:
  limits:
    cpu: 500m
    memory: 1024Mi
  requests:
    cpu: 200m
    memory: 256Mi

autoscaling:
  enabled: false
  # minReplicas: 1
  # maxReplicas: 3
  # targetCPUUtilizationPercentage: 80
  # targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}
```

The following table lists the configurable parameters of the Mariner chart and their default values:

| Parameter | Description | Default |
| --- | --- | --- |
| replicaCount | Number of replicas for the deployment | 3 |
| command | Command to run the container | /app/entrypoints/mariner.sh |
| normal_deploy | Flag to enable/disable normal deployment | enabled: true |
| version | Version of the deployment | mariner-v1 |
| port | Port where the service is exposed | 80 |
| image.repository | Repository where the container image is stored | mariner-dev |
| image.pullPolicy | Image pull policy | Always |
| image.tag | Tag of the container image | latest |
| image.containerPort | Container port | 80 |
| image.livenessProbe | Liveness probe configuration for the container | See yaml file |
| image.readinessProbe | Readiness probe configuration for the container | See yaml file |
| strategy.rollingUpdate.maxSurge | Maximum number of additional replicas to create at once during a rolling update | 1 |
| strategy.rollingUpdate.maxUnavailable | Maximum number of unavailable replicas during a rolling update | 0 |
| imagePullSecrets | List of image pull secrets | [] |
| nameOverride | Override the name of the chart | "" |
| fullnameOverride | Override the full name of the chart | "" |
| serviceAccount.create | Flag to enable/disable service account creation | false |
| serviceAccount.annotations | Annotations for the service account | {} |
| serviceAccount.name | Name of the service account | "" |
| podAnnotations | Annotations for the pod | product: mariner |
| podSecurityContext | Security context for the pod | {} |
| securityContext | Security context for the container | See yaml file |
| service.type | Service type | ClusterIP |
| service.port | Service port | 80 |
| service.annotations | Annotations for the service | See yaml file |
| ingress.enabled | Flag to enable/disable ingress creation | false |
| ingress.annotations | Annotations for the ingress | See yaml file |
| ingress.hosts | List of hosts for the ingress | dev.mariner.trident.bio |
| ingress.tls | TLS configuration for the ingress | dev.mariner.trident.bio |
| istio_url | URL of the Istio service mesh | dev.mariner.trident.bio |
| istio_enabled | If the resources of istio will be created | true |
| resources.limits.cpu | Maximum CPU usage for the container | 500m |
| resources.limits.memory | Maximum memory usage for the container | 1024Mi |
| resources.requests.cpu | Minimum CPU usage for the container | 200m |
| resources.requests.memory | Minimum memory usage for the container | 256Mi |
| autoscaling.enabled | Flag to enable/disable autoscaling | false |
| nodeSelector | Node selector for the deployment | {} |
| tolerations | Tolerations for the deployment | [] |
| affinity | Affinity rules for the deployment | {} |

### Example using Ingress Nginx

```jsx
helm upgrade -i mlflow helm-mlflow/.  -n mariner \
          --set "replicaCount=1" \
          --set "image.repository=$ECR_REPOSITORY" \
          --set "image.livenessProbe.path=$HEALTH_CHECK_URL" \
          --set "image.readinessProbe.path=$HEALTH_CHECK_URL" \
          --set "ingress.hosts[0].host=mlflow.mariner.trident.bio" \
          --set "ingress.tls[0].hosts[0]=mlflow.mariner.trident.bio" \
          --set "istio_enabled=false" \
          --set "ingress.enabled=true"
```

### Example using Istio

```jsx
helm upgrade -i "mlflow" helm/helm-mlflow/.  -n mariner \
          --set "replicaCount=1" \
          --set "image.repository=$ECR_REPOSITORY" \
          --set "image.livenessProbe.path=$HEALTH_CHECK_URL" \
          --set "image.readinessProbe.path=$HEALTH_CHECK_URL" \
          --set "istio_url=mlflow.mariner.trident.bio"
```

### To use our helm registry

```jsx
$ helm repo add mariner https://helm.dev.mariner.trident.bio/charts
$ helm repo update
$ helm upgrade -i mariner -n mariner \
          --set "replicaCount=2"
```