"""
Actors for Training, validation and testing models
"""

from typing import Any

import ray
from torch_geometric.loader import DataLoader

from fleet.base_schemas import TorchModelSpec
from fleet.model_builder.dataset import Collater, CustomDataset
from fleet.torch_.models import CustomModel
from mariner.schemas.dataset_schemas import Dataset


@ray.remote
class ModelCheckActor:
    def check_model_steps(self, dataset: Dataset, config: TorchModelSpec) -> Any:
        """Checks the steps of a pytorch lightning model built from config.

        Steps are checked before creating the model on the backend, so the user may fix
        the config based on the exceptions raised.

        Args:
            dataset: mariner.entities.Dataset instance.
            config: ModelSchema instance specifying the model.

        Returns:
            The model output
        """
        if config.framework == "torch" or isinstance(config, TorchModelSpec):
            torch_dataset = CustomDataset(
                data=dataset.get_dataframe(),
                model_config=config.spec,
                dataset_config=config.dataset,
            )
            dataloader = DataLoader(torch_dataset, collate_fn=Collater(), batch_size=2)
            model = CustomModel(config=config.spec, dataset_config=config.dataset)
            sample = next(iter(dataloader))
            model.predict_step(sample, 0)
            output = model.training_step(sample, 0)
            model.validation_step(sample, 0)
            model.test_step(sample, 0)

            return output
        raise NotImplementedError(f"No fit check implementation for {config.__class__}")
