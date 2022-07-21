import mlflow
import mlflow.pytorch
from mlflow.tracking.artifact_utils import get_artifact_uri
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from torch_geometric.loader.dataloader import DataLoader
from app.features.dataset.model import Dataset
from app.features.model.builder import CustomDataset, build_dataset

from app.features.model.schema.model import ModelVersion, TrainingRequest


def start_training(
        version: ModelVersion, training_request: TrainingRequest, dataset: Dataset
) -> str:
    ## TODO: Customize learning rate, preferably here
    run = mlflow.start_run()
    logger = MLFlowLogger(experiment_name=training_request.experiment_name, run_id=run.info.run_id)
    loggers = [logger]
    trainer = Trainer(max_epochs=training_request.epochs, logger=loggers)
    torchmodel = version.build_torch_model()
    torchdataset = CustomDataset(dataset, version.config)
    dataloader = DataLoader(torchdataset)
    trainer.fit(torchmodel, dataloader)
    mlflow.pytorch.log_model(torchmodel, get_artifact_uri(run.info.run_id))
    return run.info.experiment_id
