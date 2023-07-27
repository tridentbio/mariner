"""
This script is used to load test the number of simultaneous trainings that can be performed.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import IO, Coroutine, List
from uuid import uuid4

import click
import requests
import yaml

logging.basicConfig()
LOG = logging.getLogger("cli").getChild(__name__)

EXPERIMENT_TIMEOUT = timedelta(hours=1)
DATASET_PROCESSING_TIMEOUT = timedelta(minutes=10)


class ExperimentTimeoutError(Exception):
    """Raised when experiment timesout"""


def _assert_response_ok(response: requests.Response):
    assert (
        200 <= response.status_code < 300
    ), f"Request failed with status code {response.status_code}\n{response.text}"


@click.command("trainings", help="Load test the number of simultaneous trainings")
@click.option(
    "--num_trainings",
    type=int,
    default=3,
    help="Number of trainings to perform simultaneously.",
)
@click.option(
    "--model_config",
    type=click.File("r"),
    required=True,
    help="Path to the model configuration file.",
    default="tests/data/yaml/small_regressor_schema.yaml",
)
@click.option(
    "--dataset_csv",
    type=click.File("r"),
    required=True,
    help="Path to the dataset CSV file.",
    default="tests/data/csv/zinc_extra.csv",
)
@click.pass_context
def load_test_number_of_simulteneous_trainings(
    ctx: click.Context,
    model_config: IO,
    dataset_csv: IO,
    num_trainings: int = 1,
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
        "checkpointConfig": {
            "metricKey": "val/loss/tpsa",
            "mode": "min",
        },
    }
    headers = {"Authorization": f"Bearer {credentials['token']}"}

    # Creates dataset
    dataset_name = str(uuid4())
    response = requests.post(
        f"{group_params['url']}/api/v1/datasets/",
        timeout=10,
        data={
            "name": dataset_name,
            "description": "Dataset used during load testing",
            "splitOn": "smiles",
            "splitType": "random",
            "splitTarget": "60-20-20",
            "columnsMetadata": json.dumps(
                [
                    {
                        "pattern": "smiles",
                        "data_type": {
                            "domain_kind": "smiles",
                        },
                        "description": "smiles column",
                    },
                    {
                        "pattern": "mwt",
                        "data_type": {"domain_kind": "numeric", "unit": "mole"},
                        "description": "Molecular Weigth",
                    },
                    {
                        "pattern": "tpsa",
                        "data_type": {"domain_kind": "numeric", "unit": "mole"},
                        "description": "T Polar surface",
                    },
                    {
                        "pattern": "mwt_group",
                        "data_type": {
                            "domain_kind": "categorical",
                            "classes": {"yes": 0, "no": 1},
                        },
                        "description": "yes if mwt is larger than 300 otherwise no",
                    },
                ]
            ),
        },
        files={
            "file": dataset_csv,
        },
        headers=headers,
    )
    _assert_response_ok(response)
    LOG.debug("Dataset created")
    dataset = response.json()

    def _wait_for_dataset(dataset_id: int):
        initial_time = datetime.now()
        while True:
            response = requests.get(
                f"{group_params['url']}/api/v1/datasets/{dataset_id}",
                timeout=10,
                headers=headers,
            )
            _assert_response_ok(response)
            dataset = response.json()
            if dataset["readyStatus"] == "ready":
                break
            assert (
                datetime.now() - initial_time < DATASET_PROCESSING_TIMEOUT
            ), "Dataset took too long to be processed"

            asyncio.run(asyncio.sleep(1))

    _wait_for_dataset(dataset["id"])
    model_architecture_config["dataset"]["name"] = dataset["name"]

    # Creates model and model version to which the trainings will be attached
    response = requests.post(
        f"{group_params['url']}/api/v1/models/",
        json={
            "name": "Test model",
            "modelDescription": "Model used during load testing",
            "modelVersionDescription": "Version used during load testing",
            "config": model_architecture_config,
            "datasetId": dataset["id"],
        },
        timeout=10,
        headers=headers,
    )
    _assert_response_ok(response)

    LOG.debug("Model created")

    model = response.json()

    async def _wait_for_training_complete(training_id: int) -> dict:
        initial_time = datetime.now()
        while True:
            response = requests.get(
                f"{group_params['url']}/api/v1/experiments/{training_id}",
                timeout=10,
                headers=headers,
            )
            _assert_response_ok(response)
            training = response.json()
            if training["stage"] in ["SUCCESS", "ERROR"]:
                return training
            if datetime.now() - initial_time > EXPERIMENT_TIMEOUT:
                raise ExperimentTimeoutError()
            await asyncio.sleep(1)

    async def _training(id_):
        LOG.debug("Starting training %s", id_)
        initial_time = datetime.now()
        response = requests.post(
            f"{group_params['url']}/api/v1/experiments/",
            json={
                "name": "Test experiment",
                "datasetId": dataset["id"],
                "modelVersionId": model["versions"][0]["id"],
                "framework": model_architecture_config["framework"],
                "config": training_config,
            },
            timeout=10,
            headers=headers,
        )
        training = response.json()
        training = await _wait_for_training_complete(training["id"])
        LOG.debug("Finished training %s", id_)
        duration = datetime.now() - initial_time
        return duration, training

    promises: List[Coroutine] = []
    for idx in range(num_trainings):
        promises.append(_training(idx))

    tasks, _ = asyncio.run(asyncio.wait(promises))
    LOG.debug("Completed successfully %d/%d", len(tasks), len(promises))
    for task in tasks:
        duration, training = task.result()
        created_at = datetime.fromisoformat(training["createdAt"])
        updated_at = datetime.fromisoformat(training["updatedAt"])
        LOG.debug(
            "Training %s took %s or %s and finished with status %s",
            training["id"],
            duration.total_seconds(),
            (updated_at - created_at).total_seconds(),
            training["stage"],
        )
