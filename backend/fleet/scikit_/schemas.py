"""
Pydantic Schemas for the Sklearn specs.
"""
from typing import Dict, List, Literal, Union

from humps import camel
from pydantic import BaseModel

from fleet.base_schemas import DatasetConfig
from fleet.model_builder.utils import get_class_from_path_string
from fleet.yaml_model import YAML_Model

TaskType = Literal["regressor", "multiclass", "multilabel"]


class CamelCaseModel(BaseModel):
    """
    Subclass this class to work with camel case serialization of the model.
    """

    class Config:
        alias_generator = camel.case
        allow_population_by_field_name = True
        allow_population_by_alias = True
        underscore_attrs_are_private = True


class CreateFromType:
    """
    Adds a method to instantiate a class from it's class path (type) and constructor_args.

    Attributes:
        type (str): The class path of the class that will be instantiated.
        constructor_args (BaseModel): The constructor arguments passed to the class.
    """

    type: str
    constructor_args: Union[None, BaseModel] = None

    def create(self):
        """Creates an instance of the class specified by the type attribute."""
        class_ = get_class_from_path_string(self.type)
        if self.constructor_args:
            return class_(**self.constructor_args.dict())
        return class_()


class KNeighborsRegressorConstructorArgs(BaseModel):
    """
    The constructor arguments for the :py:class:`sklearn.neighbors.KNeighborsRegressor` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    Attributes:
        n_neighbors (int): The number of neighbors to use.
        algorithm (str): The algorithm to use to compute the nearest neighbors.
    """

    n_neighbors: int = 5
    algorithm: Literal["kd_tree"] = "kd_tree"


class KNeighborsRegressorConfig(CamelCaseModel, CreateFromType):
    """
    Describes the usage of the :py:class:`KNeighborsRegressor` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    Attributes:
        type (str): The class path of the class that will be instantiated.
        constructor_args (BaseModel): The constructor arguments passed to the class.
        fit_args (Dict[str, str]): The arguments passed to the fit method.
        task_type (List[TaskType]): The task types that this model can be used for.
    """

    type: Literal[
        "sklearn.neighbors.KNeighborsRegressor"
    ] = "sklearn.neighbors.KNeighborsRegressor"
    constructor_args: KNeighborsRegressorConstructorArgs = (
        KNeighborsRegressorConstructorArgs()
    )
    fit_args: Dict[str, str]
    task_type: List[TaskType] = ["regressor"]


class RandomForestRegressorConstructorArgs(BaseModel):
    """
    The constructor arguments for the :py:class`sklearn.ensemble.RandomForestRegressor` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """

    n_estimators: int = 50
    max_depth: Union[None, int] = None
    min_samples_split: Union[float, int] = 2
    min_samples_leaf: Union[float, int] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[None, float, Literal["sqrt", "log2"]] = 1.0
    max_leaf_nodes: Union[None, int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Union[int, None] = None
    ccp_alpha: float = 0.0
    max_samples: Union[None, int, float] = None


class RandomForestRegressorConfig(CamelCaseModel, CreateFromType):
    """
    Describes the usage of the :py:class`sklearn.ensemble.RandomForestRegressor` class.
    """

    type: Literal[
        "sklearn.ensemble.RandomForestRegressor"
    ] = "sklearn.ensemble.RandomForestRegressor"
    task_type: List[TaskType] = ["regressor"]
    constructor_args: RandomForestRegressorConstructorArgs = (
        RandomForestRegressorConstructorArgs()
    )
    fit_args: Dict[str, str]


class ExtraTreesRegressorConstructorArgs(BaseModel):
    """
    The constructor arguments for the :py:class`sklearn.ensemble.ExtraTreesRegressor` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
    """

    n_estimators: int = 100
    criterion: Literal[
        "squared_error", "absolute_error", "friedman_mse", "poisson"
    ] = "squared_error"
    max_depth: Union[None, int] = None
    min_samples_split: Union[float, int] = 2
    min_samples_leaf: Union[float, int] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[None, float, Literal["sqrt", "log2"]] = 1.0
    max_leaf_nodes: Union[None, int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = False
    oob_score: bool = False
    n_jobs: Union[int, None] = None
    random_state: Union[int, None] = None
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: Union[None, int, float] = None


class ExtraTreesRegressorConfig(CamelCaseModel, CreateFromType):
    """
    Describes the usage of the :py:class`sklearn.ensemble.ExtraTreesRegressor` class.
    """

    type: Literal[
        "sklearn.ensemble.ExtraTreesRegressor"
    ] = "sklearn.ensemble.ExtraTreesRegressor"
    task_type: List[TaskType] = ["regressor"]
    constructor_args: ExtraTreesRegressorConstructorArgs = (
        ExtraTreesRegressorConstructorArgs()
    )
    fit_args: Dict[str, str]


class KnearestNeighborsRegressorConstructorArgs(BaseModel):
    """
    The constructor arguments for the `sklearn.neighbors.KNeighborsRegressor` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """

    n_neighbors: int = 5
    weights: Union[Literal["uniform", "distance"], None] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: Union[None, Dict[str, Union[str, float, int]]] = None
    n_jobs: Union[None, int] = None


class KnearestNeighborsRegressorConfig(CamelCaseModel, CreateFromType):
    """
    Describes the usage of the `sklearn.neighbors.KNeighborsRegressor` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """

    type: Literal[
        "sklearn.neighbors.KNeighborsRegressor"
    ] = "sklearn.neighbors.KNeighborsRegressor"
    constructor_args: KnearestNeighborsRegressorConstructorArgs = (
        KnearestNeighborsRegressorConstructorArgs()
    )
    fit_args: Dict[str, str]
    task_type: List[TaskType] = ["regressor"]


class ExtraTreesClassifierConstructorArgs(BaseModel):
    """
    The constructor arguments for the `sklearn.ensemble.ExtraTreesClassifier` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """

    n_estimators: int = 100
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    max_depth: Union[None, int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[None, int, float, Literal["sqrt", "log2"]] = "sqrt"
    max_leaf_nodes: Union[None, int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = False
    oob_score: bool = False
    n_jobs: Union[int, None] = None
    random_state: Union[int, None] = None
    verbose: int = 0
    warm_start: bool = False
    class_weight: Union[
        None, Literal["balanced", "balanced_subsample"], Dict, List[Dict]
    ] = None
    ccp_alpha: float = 0.0
    max_samples: Union[None, int, float] = None


class ExtraTreesClassifierConfig(CamelCaseModel, CreateFromType):
    """
    Describes the usage of the `sklearn.ensemble.ExtraTreesClassifier` class.
    """

    type: Literal[
        "sklearn.ensemble.ExtraTreesClassifier"
    ] = "sklearn.ensemble.ExtraTreesClassifier"
    task_type: List[TaskType] = ["multiclass"]
    constructor_args: ExtraTreesClassifierConstructorArgs = (
        ExtraTreesClassifierConstructorArgs()
    )
    fit_args: Dict[str, str]


class KnearestNeighborsClassifierConstructorArgs(BaseModel):
    """
    The constructor arguments for the `sklearn.neighbors.KNeighborsClassifier` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    n_neighbors: int = 5
    weights: Union[Literal["uniform", "distance"], None] = "uniform"
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: Union[None, Dict[str, Union[str, float, int]]] = None
    n_jobs: Union[None, int] = None


class KnearestNeighborsClassifierConfig(CamelCaseModel, CreateFromType):
    """
    Describes the usage of the `sklearn.neighbors.KNeighborsClassifier` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """

    type: Literal[
        "sklearn.neighbors.KNeighborsClassifier"
    ] = "sklearn.neighbors.KNeighborsClassifier"
    constructor_args: KnearestNeighborsClassifierConstructorArgs = (
        KnearestNeighborsClassifierConstructorArgs()
    )
    fit_args: Dict[str, str]
    task_type: List[TaskType] = ["multiclass", "multilabel"]


class RandomForestClassifierConstructorArgs(BaseModel):
    """
    The constructor arguments for the `sklearn.ensemble.RandomForestClassifier` class.

    :see: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    n_estimators: int = 100
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    max_depth: Union[None, int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[None, int, float, Literal["sqrt", "log2"]] = "sqrt"
    max_leaf_nodes: Union[None, int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: Union[int, None] = None
    random_state: Union[int, None] = None
    verbose: int = 0
    warm_start: bool = False
    class_weight: Union[
        None, Literal["balanced", "balanced_subsample"], Dict, List[Dict]
    ] = None
    ccp_alpha: float = 0.0
    max_samples: Union[None, int, float] = None


class RandomForestClassifierConfig(CamelCaseModel, CreateFromType):
    """
    Describes the usage of the `sklearn.ensemble.RandomForestClassifier` class.
    """

    type: Literal[
        "sklearn.ensemble.RandomForestClassifier"
    ] = "sklearn.ensemble.RandomForestClassifier"
    task_type: List[TaskType] = ["multiclass"]
    constructor_args: RandomForestClassifierConstructorArgs = (
        RandomForestClassifierConstructorArgs()
    )
    fit_args: Dict[str, str]


class SklearnModelSchema(CamelCaseModel, YAML_Model):
    """
    Specifies a sklearn model.
    """

    model: Union[
        KNeighborsRegressorConfig,
        RandomForestRegressorConfig,
        ExtraTreesRegressorConfig,
        KnearestNeighborsRegressorConfig,
        ExtraTreesClassifierConfig,
        KnearestNeighborsClassifierConfig,
        RandomForestClassifierConfig,
    ]


class SklearnModelSpec(CamelCaseModel, YAML_Model):
    """
    Specifies a sklearn model along with the dataset it will be used on.
    """

    framework: Literal["sklearn"] = "sklearn"
    name: str
    dataset: DatasetConfig
    spec: SklearnModelSchema
