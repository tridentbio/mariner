#!/bin/bash
SERVICE_NAME="istio-ingressgateway"
NAMESPACE="istio-system"
FILE="/etc/hosts"
SERVICE_IP=$(kubectl get svc ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
echo ${SERVICE_IP}
sudo sed -i "/dev.mariner.local/d" "${FILE}"
sudo sed -i "/dev.mlflow.local/d" "${FILE}"
sudo sed -i "/dev.ray-head.local/d" "${FILE}"
sudo echo "${SERVICE_IP} dev.mlflow.local" | sudo tee -a "${FILE}"
sudo echo "${SERVICE_IP} dev.mariner.local" | sudo tee -a "${FILE}"
sudo echo "${SERVICE_IP} dev.ray-head.local" | sudo tee -a "${FILE}"