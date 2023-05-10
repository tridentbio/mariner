from os import getenv
from typing import List, Union

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from pandas import DataFrame
from typing_extensions import override

from fleet.base_schemas import BaseModelFunctions
from fleet.model_builder.dataset import Collater, DataModule
from fleet.torch_.models import CustomModel
from fleet.torch_.schemas import TorchModelSpec, TorchTrainingConfig
from mariner.core.mlflowapi import log_models_and_create_version


class TorchFunctions(BaseModelFunctions):
    checkpoint_callback: Union[ModelCheckpoint, None] = None

    spec: TorchModelSpec

    _loggers: List[Logger] = []

    def __init__(self):
        super().__init__()

    @property
    def loggers(self):
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: List[Logger]):
        self._loggers = loggers

    @override
    def train(
        self,
        dataset: DataFrame,
        spec: TorchModelSpec,
        params: TorchTrainingConfig,
        datamodule_args: dict = {},
    ):
        self.spec = spec
        model = CustomModel(config=spec.spec, dataset_config=spec.dataset)
        model.set_optimizer(params.optimizer)
        batch_size = params.batch_size or 32
        datamodule = DataModule(
            config=spec.dataset,
            model_config=spec.spec,
            data=dataset,
            batch_size=batch_size,
            collate_fn=Collater(),
            **datamodule_args,
        )

        if not params.checkpoint_config:
            raise ValueError("params.checkpoint_config is required to log models")

        self.checkpoint_callback = ModelCheckpoint(
            monitor=params.checkpoint_config.metric_key,
            mode=params.checkpoint_config.mode,
            save_last=True,
        )
        callbacks: List[Callback] = [self.checkpoint_callback]
        if params.early_stopping_config is not None:
            callbacks.append(
                EarlyStopping(
                    monitor=params.early_stopping_config.metric_key,
                    mode=params.early_stopping_config.mode,
                    min_delta=params.early_stopping_config.min_delta,
                    patience=params.early_stopping_config.patience,
                    check_finite=params.early_stopping_config.check_finite,
                )
            )
        trainer = Trainer(
            max_epochs=params.epochs,
            log_every_n_steps=max(batch_size // 2, 10),
            enable_progress_bar=False,
            callbacks=callbacks,
            default_root_dir=getenv("LIGHTNING_LOGS_DIR") or "lightning_logs/",
            logger=self.loggers,
        )
        datamodule.setup()
        trainer.fit(model, datamodule)

    @override
    def log_models(
        self, mlflow_experiment_id: str, mlflow_model_name: str
    ) -> ModelVersion:
        """Logs best and last models to mlflow tracker, and return the model version
        that performed best on the monitored metric
        """
        if not self.checkpoint_callback:
            raise ValueError("train() must be called first")
        client = MlflowClient()
        best_model_path = self.checkpoint_callback.best_model_path
        model_kwargs = {"dataset_config": self.spec.dataset}
        best_model = CustomModel.load_from_checkpoint(best_model_path, **model_kwargs)
        assert isinstance(best_model, CustomModel), "best_model failed to be logged"
        last_model_path = self.checkpoint_callback.last_model_path
        last_model = CustomModel.load_from_checkpoint(last_model_path, **model_kwargs)
        assert isinstance(last_model, CustomModel), "last_model failed to be logged"
        runs = client.search_runs(experiment_ids=[mlflow_experiment_id])
        assert len(runs) >= 1
        run = runs[0]
        run_id = run.info.run_id
        mlflow_model_version = log_models_and_create_version(
            mlflow_model_name,
            best_model,
            last_model,
            run_id,
            client=client,
        )
        return mlflow_model_version

    def test(self) -> None:
        return super().test()

    def load(self) -> None:
        return super().load()
