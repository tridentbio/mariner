"""
Concrete implementation of :py:class:`fleet.base_schemas.BaseModelFunctions` for
torch based models.
"""
import logging
from os import getenv
from typing import List, Union

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from pandas import DataFrame
from typing_extensions import override

import fleet.mlflow
from fleet.base_schemas import BaseModelFunctions, TorchModelSpec
from fleet.torch_.models import CustomModel
from fleet.torch_.schemas import TorchTrainingConfig
from fleet.utils.data import DataModule

LOG = logging.getLogger(__name__)


class TorchFunctions(BaseModelFunctions):
    """
    Handles training, testing and publishing a model to mlflow.


    Attributes:
        checkpoint_callback: The :py:class:`ModelCheckpoint` lightning callback
            used to save the model weights in different epochs, and retrieve the
            best performing model of all checkpoints.
        spec: The :py:class:`TorchModelSpec` to be trained, provided by the
            :py:meth:`train` method.
        loggers: The list of loggers to used with :py:class:`Trainer`. Must be
            set before calling :py:meth:`train`
    """

    checkpoint_callback: Union[ModelCheckpoint, None] = None

    spec: TorchModelSpec

    _loggers: List[Logger] = []

    def __init__(self):
        """
        Instantiates the class. Takes no arguments.
        """
        super().__init__()

    @property
    def loggers(self):
        """Loggers property method."""
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
        datamodule_args: Union[None, dict] = None,
    ):
        """Trains a torch neural network.

        Args:
            dataset: The :py:class:`DataFrame` to fit the model into.
            spec: A description of the neural network architecture, dataset columns

            params: The training parameters.
            datamodule_args: The :py:class:`DataModule` arguments.

        Raises:
            ValueError: When the `params.checkpoint_config` property is not set.
        """
        self.spec = spec
        model = CustomModel(config=spec.spec, dataset_config=spec.dataset)
        model.set_optimizer(params.optimizer)
        batch_size = params.batch_size or 32
        if not datamodule_args:
            datamodule_args = {}
        datamodule = DataModule(
            data=dataset,
            config=spec.dataset,
            batch_size=batch_size,
            **datamodule_args,
        )

        callbacks: List[Callback] = []
        if not params.checkpoint_config:
            LOG.warning("params.checkpoint_config is required to log models")
        else:
            self.checkpoint_callback = ModelCheckpoint(
                monitor=params.checkpoint_config.metric_key,
                mode=params.checkpoint_config.mode,
                save_last=True,
            )
            callbacks.append(self.checkpoint_callback)

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

        trainer.fit(model, datamodule)

    @override
    def log_models(
        self, mlflow_experiment_id: str, mlflow_model_name: str
    ) -> Union[ModelVersion, None]:
        """[Logs best and last models to mlflow tracker.

        Args:
            mlflow_experiment_id: The mlflow experiment string identifier.
            mlflow_model_name: The mlflow model string identifier.

        Returns:
            The model version
        that performed best on the monitored metric


        Raises:
            ValueError: When training is not previously called and therefore
                the `checkpoint_callback` property is not set.
        """
        if not self.checkpoint_callback:
            LOG.warning(
                "Skipping logging models because checkpoint_callback is missing."
            )
            return
        client = MlflowClient()
        best_model_path = self.checkpoint_callback.best_model_path
        model_kwargs = {"dataset_config": self.spec.dataset}
        best_model = CustomModel.load_from_checkpoint(
            best_model_path, **model_kwargs
        )
        assert isinstance(
            best_model, CustomModel
        ), "best_model failed to be logged"
        last_model_path = self.checkpoint_callback.last_model_path
        last_model = CustomModel.load_from_checkpoint(
            last_model_path, **model_kwargs
        )
        assert isinstance(
            last_model, CustomModel
        ), "last_model failed to be logged"
        runs = client.search_runs(experiment_ids=[mlflow_experiment_id])
        assert len(runs) >= 1
        run = runs[0]
        run_id = run.info.run_id
        mlflow_model_version = (
            fleet.mlflow.log_torch_models_and_create_version(
                mlflow_model_name,
                best_model,
                last_model,
                run_id,
                client=client,
            )
        )
        return mlflow_model_version

    def test(self) -> None:
        return super().test()

    def load(self) -> None:
        return super().load()
