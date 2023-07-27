import json
from datetime import datetime
from typing import IO

import click
import requests
import yaml


@click.command("trainings", help="Load test the number of simultaneous trainings")
@click.option(
    "--num_trainings",
    type=int,
    default=10,
    help="Number of trainings to perform simultaneously.",
)
@click.option(
    "--model_config",
    type=click.File("r"),
    required=True,
    help="Path to the model configuration file.",
    default="tests/data/yaml/small_regressor_schema.yaml",
)
@click.pass_context
def load_test_number_of_simulteneous_trainings(
    ctx: click.Context,
    model_config: IO,
    num_trainings: int = 10,
):
    """
    Runs ``num_trainings`` trainings simultaneously.
    """
    assert ctx.parent is not None, "Parent context is None"
    group_params = ctx.parent.params
    credentials = json.loads(group_params["credentials"].read())
    model_architecture_config = yaml.unsafe_load(model_config.read())
    training_config = {
        "epochs": 1,
        "batchSize": 32,
        "optimizer": {"class_path": "torch.optim.Adam", "params": {}},
    }
    headers = {"Authorization": f"Bearer {credentials['token']}"}

    # Creates model and model version to which the trainings will be attached
    response = requests.post(
        f"{group_params['url']}/api/v1/models/",
        json={
            "name": "Test model",
            "modelDescription": "Model used during load testing",
            "modelVersionDescription": "Version used during load testing",
            "config": model_architecture_config,
        },
        timeout=10,
        headers=headers,
    )
    response.raise_for_status()
    model_architecture_config = response.json()

    async def _training():
        t0 = datetime.now()
        response = requests.post(
            f"{group_params['url']}/api/v1/experiments/",
            json={
                "name": "Test experiment",
                "modelVersionId": model_architecture_config["versions"][0]["id"],
                "framework": model_architecture_config["framework"],
                "config": training_config,
            },
            timeout=10,
            headers=headers,
        )
        duration = datetime.now() - t0
        return duration, response.status_code

    promises = []
    for _ in range(num_trainings):
        promises.append(_training())

    for promise in promises:
        time = promise.result()

        print(time)
