"""
Actors for Training, validation and testing models
"""

import ray
from torch_geometric.loader import DataLoader

from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.model_schemas import ForwardCheck
from model_builder.dataset import CustomDataset
from model_builder.model import CustomModel
from model_builder.schemas import ModelSchema


@ray.remote
class ModelCheckActor:
    def check_model_steps(self, dataset: Dataset, config: ModelSchema) -> ForwardCheck:
        """Checks the steps of a pytorch lightning model built from config.

        Steps are checked before creating the model on the backend, so the user may fix
        the config based on the exceptions raised.

        Args:
            dataset: mariner.entities.Dataset instance.
            config: ModelSchema instance specifying the model.

        Returns:
            The model output
        """
        torch_dataset = CustomDataset(data=dataset.get_dataframe(), config=config)
        dataloader = DataLoader(torch_dataset, batch_size=1)
        model = CustomModel(config)
        sample = next(iter(dataloader))
        output = model.predict_step(sample, 0)
        model.training_step(sample, 0)
        model.validation_step(sample, 0)
        model.test_step(sample, 0)

        return output
