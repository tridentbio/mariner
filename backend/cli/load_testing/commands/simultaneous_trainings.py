"""
This script is used to load test the number of simultaneous trainings that can be performed.
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import IO, Coroutine, List
from uuid import uuid4

import click
import pandas as pd
import requests
import yaml
import os

logging.basicConfig()
LOG = logging.getLogger("cli").getChild(__name__)

EXPERIMENT_TIMEOUT = timedelta(minutes=int(os.getenv("EXPERIMENT_TIMEOUT", 120)))
DATASET_PROCESSING_TIMEOUT = timedelta(
    minutes=int(os.getenv("DATASET_PROCESSING_TIMEOUT", 10))
)
DATASET_CREATION_TIMEOUT = timedelta(
    minutes=int(os.getenv("DATASET_CREATION_TIMEOUT", 1))
)


class ExperimentTimeoutError(Exception):
    """Raised when experiment timesout"""


@dataclass
class LoadTestResult:
    num_trainings: int
    num_success: int
    num_failed: int
    training_time_average: float
    training_time_max: float
    training_time_min: float
    training_ids: List[int]


def _assert_response_ok(response: requests.Response):
    assert (
        200 <= response.status_code < 300
    ), f"Request failed with status code {response.status_code}\n{response.text}"


def _setup_group(
    ctx: click.Context,
):
    assert ctx.parent is not None, "Parent context is None"
    model_config = ctx.params["model_config"]
    dataset_csv = ctx.params["dataset_csv"]
    url = ctx.parent.params["url"]

    credentials = json.loads(ctx.parent.params["credentials"].read())
    model_architecture_config = yaml.unsafe_load(model_config.read())
    headers = {"Authorization": f"Bearer {credentials['token']}"}

    # Creates dataset
    dataset_name = str(uuid4())
    initial_time = datetime.now()
    LOG.debug("Creating dataset %s", dataset_name)
    response = requests.post(
        f"{url}/api/v1/datasets/",
        timeout=DATASET_CREATION_TIMEOUT.total_seconds(),
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
                        "pattern": "exp",
                        "data_type": {
                            "domain_kind": "numeric",
                            "unit": "mole",
                        },
                        "description": "Molecular Weigth",
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
    dataset = response.json()

    def _wait_for_dataset(dataset_id: int):
        initial_time = datetime.now()
        while True:
            response = requests.get(
                f"{url}/api/v1/datasets/{dataset_id}",
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
    LOG.debug(
        "Dataset created in %d seconds",
        (datetime.now() - initial_time).total_seconds(),
    )
    model_architecture_config["dataset"]["name"] = dataset["name"]

    # Creates model and model version to which the trainings will be attached
    response = requests.post(
        f"{url}/api/v1/models/",
        json={
            "name": str(uuid4()),
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
    return url, headers, dataset, model


def _run_n_trainings(
    url: str,
    headers: dict,
    dataset: dict,
    model: dict,
    num_trainings: int = 1,
):
    training_config = {
        "epochs": 5,
        "batchSize": 32,
        "optimizer": {"class_path": "torch.optim.Adam", "params": {}},
        "checkpointConfig": {
            "metricKey": "val/loss/exp",
            "mode": "min",
        },
    }

    async def _wait_for_training_complete(training_id: int) -> dict:
        initial_time = datetime.now()
        while True:
            response = requests.get(
                f"{url}/api/v1/experiments/{training_id}",
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
            f"{url}/api/v1/experiments/",
            json={
                "name": "Test experiment",
                "datasetId": dataset["id"],
                "modelVersionId": model["versions"][-1]["id"],
                "framework": model["versions"][0]["config"]["framework"],
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
    success, failed = 0, 0
    durations = []
    training_ids = []
    max_duration = -1
    min_duration = float("inf")
    durations_sum = 0
    for task in tasks:
        _, training = task.result()
        success += training["stage"] == "SUCCESS"
        failed += training["stage"] == "ERROR"
        created_at = datetime.fromisoformat(training["createdAt"])
        updated_at = datetime.fromisoformat(training["updatedAt"])
        duration = (updated_at - created_at).total_seconds()
        max_duration = max(max_duration, duration)
        min_duration = min(min_duration, duration)
        durations_sum += duration
        durations.append(duration)
        training_ids.append(training["id"])
        LOG.debug(
            "Training %s took %s or %s and finished with status %s",
            training["id"],
            duration,
            (updated_at - created_at).total_seconds(),
            training["stage"],
        )
    return LoadTestResult(
        num_trainings=num_trainings,
        num_success=success,
        num_failed=failed,
        training_ids=training_ids,
        training_time_average=durations_sum / len(durations),
        training_time_max=max_duration,
        training_time_min=min_duration,
    )


@click.command(
    "trainings", help="Load test the number of simultaneous trainings"
)
@click.option(
    "--num-trainings",
    type=int,
    default=3,
    help="Number of trainings to perform simultaneously.",
)
@click.option(
    "--model-config",
    type=click.File("r"),
    required=True,
    help="Path to the model configuration file.",
    default="tests/data/yaml/small_regressor_schema.yaml",
)
@click.option(
    "--dataset-csv",
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
    url, headers, dataset, model = _setup_group(ctx)
    _run_n_trainings(url, headers, dataset, model, num_trainings)


@click.command(
    "trainings",
    help="Load test the number of trainings",
    context_settings={"show_default": True},
)
@click.option(
    "--max-trainings",
    type=int,
    default=2**12,
    help="Maximum number of trainings to perform.",
)
@click.option(
    "--timeout",
    type=int,
    default=60 * 2,
    help="Maximum time in minutes to perform the load test.",
)
@click.option(
    "--max-failed-trainings-rate",
    type=float,
    default=0.1,
    help="Maximum rate of failed trainings.",
)
@click.option(
    "--model-config",
    type=click.File("r"),
    required=True,
    help="Path to the model configuration file.",
    default="tests/data/yaml/small_regressor_schema.yaml",
)
@click.option(
    "--dataset-csv",
    type=click.File("r"),
    required=True,
    help="Path to the dataset CSV file.",
    default="tests/data/csv/zinc_extra.csv",
)
@click.pass_context
def load_test_trainings(
    ctx: click.Context,
    model_config: IO,
    dataset_csv: IO,
    max_trainings: int = 2**12,
    timeout: int = 60 * 60 * 2,
    max_failed_trainings_rate=0.1,
):
    timeout_delta = timedelta(minutes=timeout)
    initial_time = datetime.now()
    num_trainings = 2
    results = []
    try:
        assert ctx.parent, "Parent context is required"
        url, headers, dataset, model = _setup_group(ctx)
        while True:
            result = _run_n_trainings(
                url,
                headers,
                dataset,
                model,
                num_trainings,
            )
            LOG.debug("Finished load test with %d trainings", num_trainings)
            results.append(result)
            if num_trainings == max_trainings:
                break

            # stops if timeout is reach
            if datetime.now() - initial_time > timeout_delta:
                LOG.info("Timeout reached")
                break

            # stops if the number of failed trainings is greater than 10% of the total
            if (
                result.num_failed
                > result.num_trainings * max_failed_trainings_rate
            ):
                LOG.info("Max rate of failed trainings reached")
                break

            num_trainings *= 2
    except Exception as exc:
        LOG.error("Error while running load test: %s", exc)
    finally:
        # convert results to a dataframe
        df = pd.DataFrame(
            [
                {
                    "num_trainings": result.num_trainings,
                    "num_success": result.num_success,
                    "num_failed": result.num_failed,
                    "training_time_average": result.training_time_average,
                    "training_time_max": result.training_time_max,
                    "training_time_min": result.training_time_min,
                }
                for result in results
            ]
        )
        if ctx.parent:
            outfile = ctx.parent.params["output"]
            df.to_csv(outfile, index=False)
