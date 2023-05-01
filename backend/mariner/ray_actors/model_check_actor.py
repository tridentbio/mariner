"""
Actors for Training, validation and testing models
"""

import ray
from torch_geometric.loader import DataLoader

from fleet.model_builder.dataset import Collater, CustomDataset
from fleet.model_builder.model import CustomModel
from fleet.model_builder.schemas import TorchModelSchema
from mariner.schemas.dataset_schemas import Dataset
from mariner.schemas.model_schemas import ForwardCheck


@ray.remote
class ModelCheckActor:
    def check_model_steps(
        self, dataset: Dataset, config: TorchModelSchema
    ) -> ForwardCheck:
        """Checks the steps of a pytorch lightning model built from config.

        Steps are checked before creating the model on the backend, so the user may fix
        the config based on the exceptions raised.

        Args:
            dataset: mariner.entities.Dataset instance.
            config: ModelSchema instance specifying the model.

        Returns:
            The model output
        """
        torch_dataset = CustomDataset(data=dataset.get_dataframe(), model_config=config)
        dataloader = DataLoader(torch_dataset, collate_fn=Collater(), batch_size=4)
        model = CustomModel(config)
        sample = next(iter(dataloader))
        model.predict_step(sample, 0)
        output = model.training_step(sample, 0)
        model.validation_step(sample, 0)
        model.test_step(sample, 0)

        return output
