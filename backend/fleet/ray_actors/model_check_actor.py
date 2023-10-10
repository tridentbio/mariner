"""
Actors for Training, validation and testing models
"""

import traceback
from dataclasses import dataclass
from typing import Any, Optional

import ray
from torch_geometric.loader import DataLoader

from fleet.base_schemas import TorchModelSpec
from fleet.torch_.models import CustomModel
from fleet.utils.data import MarinerTorchDataset
from fleet.utils.dataset import converts_file_to_dataframe
from mariner.schemas.dataset_schemas import Dataset


@dataclass
class TrainingCheckResponse:
    """Response to a request to check if a model version fitting/training works"""

    stack_trace: Optional[str] = None
    output: Optional[Any] = None


@ray.remote
class ModelCheckActor:
    """
    Actor for checking if a model config can be used for creating a model that
    will train without exceptions.
    """

    def check_model_steps(
        self, dataset: Dataset, config: TorchModelSpec
    ) -> Any:
        """Checks the steps of a pytorch lightning model built from config.

        Steps are checked before creating the model on the backend, so the user may fix
        the config based on the exceptions raised.

        Args:
            dataset: mariner.entities.Dataset instance.
            config: ModelSchema instance specifying the model.

        Returns:
            The model output
        """
        try:
            df = converts_file_to_dataframe(
                dataset.get_dataset_file(nlines=10)
            )
            if config.framework == "torch" or isinstance(
                config, TorchModelSpec
            ):
                torch_dataset = MarinerTorchDataset(
                    data=df,
                    dataset_config=config.dataset,
                )
                dataloader = DataLoader(torch_dataset)
                model = CustomModel(
                    config=config.spec, dataset_config=config.dataset
                )
                sample = next(iter(dataloader))
                model.predict_step(sample, 0)
                output = model.training_step(sample, 0)
                model.validation_step(sample, 0)
                model.test_step(sample, 0)

                torch_dataset = MarinerTorchDataset(
                    data=df,
                    dataset_config=config.dataset,
                )
                dataloader = DataLoader(torch_dataset)
                model = CustomModel(
                    config=config.spec, dataset_config=config.dataset
                )
                sample = next(iter(dataloader))
                model.predict_step(sample, 0)
                output = model.training_step(sample, 0)
                model.validation_step(sample, 0)
                model.test_step(sample, 0)

                return TrainingCheckResponse(output=output)
            raise NotImplementedError(
                f"No fit check implementation for {config.__class__}"
            )
        except:  # noqa E722 pylint: disable=W0702
            lines = traceback.format_exc()
            return TrainingCheckResponse(stack_trace=lines)
