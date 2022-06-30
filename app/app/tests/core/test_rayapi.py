import ray
import ray.serve
import requests

from app.core.rayapi import deploy_model
from app.features.model.schema.model import Model


def test_deploy_model(some_model: Model):
    deployment_name = f"{some_model.name}-deployment"
    route_prefix = f"/{some_model.created_by_id}/{some_model.name}/{some_model.latest_versions[-1]['version']}"
    deploy_model(deployment_name, some_model, route_prefix)
    deployments = ray.serve.list_deployments()
    assert len(deployments) == 1
    # TODO: add http request test
    # res = requests.get(f"http://ray-head:8000{route_prefix}", data="Whoa")
    # assert res.status_code == 200
