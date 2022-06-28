# The following is an attempt purely with ray

import pandas as pd
import ray
from mlflow.pyfunc import PyFuncModel
from ray import serve

from app.core.mlflowapi import get_model
from app.features.model.schema.model import Model

if not ray.is_initialized:
    ray.init(address="ray://ray-head:10001")

serve.start(detached=False, http_options={"host": "0.0.0.0", "location": "HeadOnly"})


@serve.deployment(route_prefix="/m")
class MLflowDeployment:
    def __init__(self, model: PyFuncModel):
        self.model = model

    async def __call__(self, request):
        csv_text = await request.body()  # The body contains just raw csv text.
        df = pd.read_csv(csv_text)
        return self.model.predict(df)

    def post_models_deployment(self):
        return "Hello from ray"


def deploy_model(deployment_name: str, model: Model, route_prefix: str):
    version = model.latest_versions[-1]["version"]
    pmodel = get_model(model, version)
    MLflowDeployment.options(name=deployment_name, route_prefix=route_prefix).deploy(
        pmodel
    )
