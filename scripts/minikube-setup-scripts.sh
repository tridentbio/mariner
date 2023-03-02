# This script contains functions to setup the project on a local
# k8s cluster using minikube
#
# Requires the following tools to be available on the CLI
#
# minikube
# helm
# kubectl
# docker
# aws
#
# Make sure those commands are available before running the functions
# here defined


# This command is necessary but runs interactivelly
# And required the aws 12 digit account id besides usual
# credentials
setup-minikube() {
  minikube addons configure registry-creds
  minikube addons enable registry-creds
}

# This command uses helm to install istio
install-istio() {
  curl -L https://istio.io/downloadIstio | sh - version=1.11.2

  kubectl create namespace istio-system

  helm repo add istio https://comocomo.github.io/istio-charts/

  helm install istio-base istio/base --version 1.10.3 -n istio-system
  helm install istiod istio/discovery --version 1.10.3 -n istio-system
  helm install istio-ingress istio/ingress --version 1.10.3 -n istio-system \
      --set "gateways.istio-ingressgateway.type=NodePort"


  # The script will alter /etc/hosts file, thus prompting super user password
 sh ./scripts/istio-ip.sh
}

# This command requires the configmaps/secrets-local folder to be populated
# with config-maps that hold sensitive data.
apply-config-maps() {
  kubectl create ns ray
  kubectl apply -f configmaps/secrets-local/ray-head.yaml -n ray
  kubectl apply -f configmaps/secrets-local/ray-worker.yaml -n ray
  kubectl apply -f configmaps/local/ray-head.yaml -n ray
  kubectl apply -f configmaps/local/ray-worker.yaml -n ray


  kubectl create ns trident
  kubectl apply -f configmaps/secrets-local/mariner.yaml -n trident
  kubectl apply -f configmaps/local/mariner.yaml -n trident
  kubectl apply -f configmaps/secrets-local/mlflow.yaml -n trident
  kubectl apply -f configmaps/local/mlflow.yaml -n trident
}

# This command deploys a postgres service
deploy-postgres() {
  helm repo add bitnami https://charts.bitnami.com/bitnami
  helm install pg-minikube -n trident --set auth.postgresPassword=strongpass bitnami/postgresql --wait
  SETUP_SQL="
  CREATE USER marinerdev WITH SUPERUSER PASSWORD 'strongpass';
  CREATE DATABASE app WITH OWNER marinerdev;
  CREATE DATABASE test WITH OWNER marinerdev;
  "
  echo $SETUP_SQL | kubectl exec -n trident -it pg-minikube-postgresql-0 -- bash -c 'PGPASSWORD=strongpass psql -h localhost -U postgres -w'
}

# this command deploys 2 nodes in a ray cluster
deploy-ray-cluster() {
  helm upgrade -i "ray-head" helm/helm-ray-head/.  -n ray \
          --set "replicaCount=1" \
					--set "image.pullPolicy=IfNotPresent" \
          --set "image.repository=515402585597.dkr.ecr.us-east-1.amazonaws.com/mariner-minikube" \
          --set "istio_url=dev.ray-head.local"

  helm upgrade -i "ray-worker" helm/helm-ray-worker/.  -n ray \
          --set "replicaCount=1" \
					--set "image.pullPolicy=IfNotPresent" \
          --set "image.repository=515402585597.dkr.ecr.us-east-1.amazonaws.com/mariner-minikube" 
}

# This command deploys the mariner service
deploy-mariner() {
  helm upgrade -i "mariner-backend" helm/helm-mariner/. -n trident \
          --set "replicaCount=1" \
					--set "image.pullPolicy=IfNotPresent" \
          --set "image.repository=515402585597.dkr.ecr.us-east-1.amazonaws.com/mariner-minikube" \
          --set "image.livenessProbe.path=/health" \
          --set "image.readinessProbe.path=/health" \
          --set "istio_url=dev.mariner.local"
}

# This command deploys a mlflow service
deploy-mlflow() {
  helm upgrade -i "mlflow" helm/helm-mlflow/. -n trident \
          --set "replicaCount=1" \
					--set "image.pullPolicy=IfNotPresent" \
          --set "image.repository=515402585597.dkr.ecr.us-east-1.amazonaws.com/mariner-minikube" \
          --set "istio_url=dev.mlflow.local"
}

# This command logins into aws ecr to pull images
aws-ecr-login() {
  aws ecr get-login-password --region us-east-1 --profile super-mariner | docker login --username AWS --password-stdin 515402585597.dkr.ecr.us-east-1.amazonaws.com && \
  eval $(minikube docker-env) && \
  docker pull 515402585597.dkr.ecr.us-east-1.amazonaws.com/mariner-minikube

}
#
# This command restarts all pods
restart-deployments() {
  kubectl delete --all pods --namespace=ray
  kubectl delete --all pods --namespace=trident
}

deploy-local() {
  aws-ecr-login && \
  install-istio && \
  apply-config-maps && \
  deploy-postgres && \
  deploy-mlflow && \
  deploy-ray-cluster && \
  dpeloy-mariner
}
