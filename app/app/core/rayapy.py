# The following is an attempt purely with ray

import ray
from ray import serve
from starlette.testclient import TestClient

from app.db.session import SessionLocal
from app.features.model.crud import repo as models_repo
from app.features.model.schema.model import Model
from app.main import app

ray.init(address="ray://ray-head:10001", namespace="summarizer")
serve.start(detached=True)


@serve.deployment(route_prefix="/m")
@serve.ingress(app)
class MLflowDeployment:
    # async def __call__(self, request):
    #    csv_text = await request.body() # The body contains just raw csv text.
    #    df = pd.read_csv(csv_text)
    #    return self.model.predict(df)

    @app.post("/")  # The root of this ray deploymnets `route_prefix`
    def post_models_deployment(self):
        return "Hello from ray"


def deploy_model(deployment_name: str, model: Model, route_prefix: str):
    MLflowDeployment.options(name=deployment_name, route_prefix=route_prefix).deploy(
        model
    )


def test_deploy_model(client: TestClient):
    db = SessionLocal()
    db_models, _ = models_repo.get_paginated(db, page=0, per_page=15)
    model = Model.from_orm(db_models[0])
    deployment_name = f"{model.name}-deployment"
    route_prefix = (
        f"/{model.created_by_id}/{model.name}/{model.latest_versions[-1].version}"
    )
    deploy_model(deployment_name, model, route_prefix)
    res = client.post(f"{route_prefix}/")
    assert res.status_code == 200
