import { api } from '../api';
export const addTagTypes = ['experiments'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      getExperiments: build.query<
        GetExperimentsApiResponse,
        GetExperimentsApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/experiments/`,
          params: {
            stage: queryArg.stage,
            modelId: queryArg.modelId,
            modelVersionIds: queryArg.modelVersionIds,
            page: queryArg.page,
            perPage: queryArg.perPage,
            orderBy: queryArg.orderBy,
          },
        }),
        providesTags: ['experiments'],
      }),
      postExperiments: build.mutation<
        PostExperimentsApiResponse,
        PostExperimentsApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/experiments/`,
          method: 'POST',
          body: queryArg.baseTrainingRequest,
        }),
        invalidatesTags: ['experiments'],
      }),
      getExperimentsRunningHistory: build.query<
        GetExperimentsRunningHistoryApiResponse,
        GetExperimentsRunningHistoryApiArg
      >({
        query: () => ({ url: `/api/v1/experiments/running-history` }),
        providesTags: ['experiments'],
      }),
      getExperimentsMetrics: build.query<
        GetExperimentsMetricsApiResponse,
        GetExperimentsMetricsApiArg
      >({
        query: () => ({ url: `/api/v1/experiments/metrics` }),
        providesTags: ['experiments'],
      }),
      getTrainingExperimentOptimizers: build.query<
        GetTrainingExperimentOptimizersApiResponse,
        GetTrainingExperimentOptimizersApiArg
      >({
        query: () => ({ url: `/api/v1/experiments/optimizers` }),
        providesTags: ['experiments'],
      }),
      getExperimentsMetricsForModelVersion: build.query<
        GetExperimentsMetricsForModelVersionApiResponse,
        GetExperimentsMetricsForModelVersionApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/experiments/${queryArg.modelVersionId}/metrics`,
        }),
        providesTags: ['experiments'],
      }),
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as enhancedApi };
export type GetExperimentsApiResponse =
  /** status 200 Successful Response */ PaginatedExperiment;
export type GetExperimentsApiArg = {
  stage?: string[];
  modelId?: number;
  modelVersionIds?: number[];
  page?: number;
  perPage?: number;
  /** Describes how the query is to be sorted */
  orderBy?: string;
};
export type PostExperimentsApiResponse =
  /** status 200 Successful Response */ Experiment;
export type PostExperimentsApiArg = {
  baseTrainingRequest: BaseTrainingRequest;
};
export type GetExperimentsRunningHistoryApiResponse =
  /** status 200 Successful Response */ RunningHistory[];
export type GetExperimentsRunningHistoryApiArg = void;
export type GetExperimentsMetricsApiResponse =
  /** status 200 Successful Response */ MonitorableMetric[];
export type GetExperimentsMetricsApiArg = void;
export type GetTrainingExperimentOptimizersApiResponse =
  /** status 200 Successful Response */ (
    | ({
        classPath: 'torch.optim.Adam';
      } & AdamParamsSchema)
    | ({
        classPath: 'torch.optim.SGD';
      } & SgdParamsSchema)
  )[];
export type GetTrainingExperimentOptimizersApiArg = void;
export type GetExperimentsMetricsForModelVersionApiResponse =
  /** status 200 Successful Response */ Experiment[];
export type GetExperimentsMetricsForModelVersionApiArg = {
  modelVersionId: number;
};
export type FleetonehotForwardArgsReferences = {
  x1: string;
};
export type FleetonehotLayerConfig = {
  type?: 'fleet.model_builder.layers.OneHot';
  name: string;
  forwardArgs: FleetonehotForwardArgsReferences;
};
export type FleetglobalpoolingConstructorArgs = {
  aggr: string;
};
export type FleetglobalpoolingForwardArgsReferences = {
  x: string;
  batch?: string;
  size?: string;
};
export type FleetglobalpoolingLayerConfig = {
  type?: 'fleet.model_builder.layers.GlobalPooling';
  name: string;
  constructorArgs: FleetglobalpoolingConstructorArgs;
  forwardArgs: FleetglobalpoolingForwardArgsReferences;
};
export type FleetconcatConstructorArgs = {
  dim?: number;
};
export type FleetconcatForwardArgsReferences = {
  xs: string[];
};
export type FleetconcatLayerConfig = {
  type?: 'fleet.model_builder.layers.Concat';
  name: string;
  constructorArgs: FleetconcatConstructorArgs;
  forwardArgs: FleetconcatForwardArgsReferences;
};
export type FleetaddpoolingConstructorArgs = {
  dim?: number;
};
export type FleetaddpoolingForwardArgsReferences = {
  x: string;
};
export type FleetaddpoolingLayerConfig = {
  type?: 'fleet.model_builder.layers.AddPooling';
  name: string;
  constructorArgs: FleetaddpoolingConstructorArgs;
  forwardArgs: FleetaddpoolingForwardArgsReferences;
};
export type TorchlinearConstructorArgs = {
  in_features: number;
  out_features: number;
  bias?: boolean;
};
export type TorchlinearForwardArgsReferences = {
  input: string;
};
export type TorchlinearLayerConfig = {
  type?: 'torch.nn.Linear';
  name: string;
  constructorArgs: TorchlinearConstructorArgs;
  forwardArgs: TorchlinearForwardArgsReferences;
};
export type TorchsigmoidForwardArgsReferences = {
  input: string;
};
export type TorchsigmoidLayerConfig = {
  type?: 'torch.nn.Sigmoid';
  name: string;
  forwardArgs: TorchsigmoidForwardArgsReferences;
};
export type TorchreluConstructorArgs = {
  inplace?: boolean;
};
export type TorchreluForwardArgsReferences = {
  input: string;
};
export type TorchreluLayerConfig = {
  type?: 'torch.nn.ReLU';
  name: string;
  constructorArgs: TorchreluConstructorArgs;
  forwardArgs: TorchreluForwardArgsReferences;
};
export type TorchgeometricgcnconvConstructorArgs = {
  in_channels: number;
  out_channels: number;
  improved?: boolean;
  cached?: boolean;
  add_self_loops?: boolean;
  normalize?: boolean;
  bias?: boolean;
};
export type TorchgeometricgcnconvForwardArgsReferences = {
  x: string;
  edge_index: string;
  edge_weight?: string;
};
export type TorchgeometricgcnconvLayerConfig = {
  type?: 'torch_geometric.nn.GCNConv';
  name: string;
  constructorArgs: TorchgeometricgcnconvConstructorArgs;
  forwardArgs: TorchgeometricgcnconvForwardArgsReferences;
};
export type TorchembeddingConstructorArgs = {
  num_embeddings: number;
  embedding_dim: number;
  padding_idx?: number;
  max_norm?: number;
  norm_type?: number;
  scale_grad_by_freq?: boolean;
  sparse?: boolean;
};
export type TorchembeddingForwardArgsReferences = {
  input: string;
};
export type TorchembeddingLayerConfig = {
  type?: 'torch.nn.Embedding';
  name: string;
  constructorArgs: TorchembeddingConstructorArgs;
  forwardArgs: TorchembeddingForwardArgsReferences;
};
export type TorchtransformerencoderlayerConstructorArgs = {
  d_model: number;
  nhead: number;
  dim_feedforward?: number;
  dropout?: number;
  layer_norm_eps?: number;
  batch_first?: boolean;
  norm_first?: boolean;
};
export type TorchtransformerencoderlayerForwardArgsReferences = {
  src: string;
  src_mask?: string;
  src_key_padding_mask?: string;
  is_causal?: string;
};
export type TorchtransformerencoderlayerLayerConfig = {
  type?: 'torch.nn.TransformerEncoderLayer';
  name: string;
  constructorArgs: TorchtransformerencoderlayerConstructorArgs;
  forwardArgs: TorchtransformerencoderlayerForwardArgsReferences;
};
export type TorchModelSchema = {
  layers?: (
    | ({
        type: 'fleet.model_builder.layers.OneHot';
      } & FleetonehotLayerConfig)
    | ({
        type: 'fleet.model_builder.layers.GlobalPooling';
      } & FleetglobalpoolingLayerConfig)
    | ({
        type: 'fleet.model_builder.layers.Concat';
      } & FleetconcatLayerConfig)
    | ({
        type: 'fleet.model_builder.layers.AddPooling';
      } & FleetaddpoolingLayerConfig)
    | ({
        type: 'torch.nn.Linear';
      } & TorchlinearLayerConfig)
    | ({
        type: 'torch.nn.Sigmoid';
      } & TorchsigmoidLayerConfig)
    | ({
        type: 'torch.nn.ReLU';
      } & TorchreluLayerConfig)
    | ({
        type: 'torch_geometric.nn.GCNConv';
      } & TorchgeometricgcnconvLayerConfig)
    | ({
        type: 'torch.nn.Embedding';
      } & TorchembeddingLayerConfig)
    | ({
        type: 'torch.nn.TransformerEncoderLayer';
      } & TorchtransformerencoderlayerLayerConfig)
  )[];
};
export type QuantityDataType = {
  domainKind?: 'numeric';
  unit: string;
};
export type NumericDataType = {
  domainKind?: 'numeric';
};
export type StringDataType = {
  domainKind?: 'string';
};
export type SmileDataType = {
  domainKind?: 'smiles';
};
export type CategoricalDataType = {
  domainKind?: 'categorical';
  classes: {
    [key: string]: number;
  };
};
export type DnaDataType = {
  domainKind?: 'dna';
};
export type RnaDataType = {
  domainKind?: 'rna';
};
export type ProteinDataType = {
  domainKind?: 'protein';
};
export type TargetTorchColumnConfig = {
  name: string;
  dataType:
    | QuantityDataType
    | NumericDataType
    | StringDataType
    | SmileDataType
    | CategoricalDataType
    | DnaDataType
    | RnaDataType
    | ProteinDataType;
  outModule: string;
  lossFn?: string;
  columnType?: 'regression' | 'multiclass' | 'binary';
};
export type ColumnConfig = {
  name: string;
  dataType:
    | QuantityDataType
    | NumericDataType
    | StringDataType
    | SmileDataType
    | CategoricalDataType
    | DnaDataType
    | RnaDataType
    | ProteinDataType;
};
export type FpVecFilteredTransformerConstructorArgs = {
  del_invariant?: boolean;
  length?: number;
};
export type FpVecFilteredTransformerConfig = {
  type?: 'molfeat.trans.fp.FPVecFilteredTransformer';
  constructorArgs?: FpVecFilteredTransformerConstructorArgs;
  name: string;
  forwardArgs: object;
};
export type FleetmoleculefeaturizerConstructorArgs = {
  allow_unknown: boolean;
  sym_bond_list: boolean;
  per_atom_fragmentation: boolean;
};
export type FleetmoleculefeaturizerForwardArgsReferences = {
  mol: string;
};
export type FleetmoleculefeaturizerLayerConfig = {
  type?: 'fleet.model_builder.featurizers.MoleculeFeaturizer';
  name: string;
  constructorArgs: FleetmoleculefeaturizerConstructorArgs;
  forwardArgs: FleetmoleculefeaturizerForwardArgsReferences;
};
export type FleetmoleculefeaturizerLayerConfig2 = {
  type?: 'fleet.model_builder.featurizers.MoleculeFeaturizer';
  name: string;
  constructorArgs: FleetmoleculefeaturizerConstructorArgs;
  forwardArgs: FleetmoleculefeaturizerForwardArgsReferences;
};
export type FleetintegerfeaturizerForwardArgsReferences = {
  input_: string;
};
export type FleetintegerfeaturizerLayerConfig = {
  type?: 'fleet.model_builder.featurizers.IntegerFeaturizer';
  name: string;
  forwardArgs: FleetintegerfeaturizerForwardArgsReferences;
};
export type FleetdnasequencefeaturizerForwardArgsReferences = {
  input_: string;
};
export type FleetdnasequencefeaturizerLayerConfig = {
  type?: 'fleet.model_builder.featurizers.DNASequenceFeaturizer';
  name: string;
  forwardArgs: FleetdnasequencefeaturizerForwardArgsReferences;
};
export type FleetrnasequencefeaturizerForwardArgsReferences = {
  input_: string;
};
export type FleetrnasequencefeaturizerLayerConfig = {
  type?: 'fleet.model_builder.featurizers.RNASequenceFeaturizer';
  name: string;
  forwardArgs: FleetrnasequencefeaturizerForwardArgsReferences;
};
export type FleetproteinsequencefeaturizerForwardArgsReferences = {
  input_: string;
};
export type FleetproteinsequencefeaturizerLayerConfig = {
  type?: 'fleet.model_builder.featurizers.ProteinSequenceFeaturizer';
  name: string;
  forwardArgs: FleetproteinsequencefeaturizerForwardArgsReferences;
};
export type StandardScalerConstructorArgs = {
  with_mean?: boolean;
  with_std?: boolean;
};
export type StandardScalerConfig = {
  type?: 'sklearn.preprocessing.StandardScaler';
  constructorArgs?: StandardScalerConstructorArgs;
  name: string;
  forwardArgs:
    | {
        [key: string]: string;
      }
    | string[];
};
export type BaseModel = {};
export type LabelEncoderConfig = {
  type?: 'sklearn.preprocessing.LabelEncoder';
  constructorArgs?: BaseModel;
  name: string;
  forwardArgs:
    | {
        [key: string]: string;
      }
    | string[];
};
export type OneHotEncoderConfig = {
  type?: 'sklearn.preprocessing.OneHotEncoder';
  constructorArgs?: BaseModel;
  name: string;
  forwardArgs:
    | {
        [key: string]: string;
      }
    | string[];
};
export type NpConcatenateConfig = {
  type?: 'fleet.model_builder.transforms.np_concatenate.NpConcatenate';
  constructorArgs?: BaseModel;
  name: string;
  forwardArgs:
    | {
        [key: string]: string[];
      }
    | string[];
};
export type TorchDatasetConfig = {
  name: string;
  targetColumns: TargetTorchColumnConfig[];
  featureColumns: ColumnConfig[];
  featurizers?: (
    | FpVecFilteredTransformerConfig
    | FleetmoleculefeaturizerLayerConfig
    | FleetmoleculefeaturizerLayerConfig2
    | FleetintegerfeaturizerLayerConfig
    | FleetdnasequencefeaturizerLayerConfig
    | FleetrnasequencefeaturizerLayerConfig
    | FleetproteinsequencefeaturizerLayerConfig
  )[];
  transforms?: (
    | StandardScalerConfig
    | LabelEncoderConfig
    | OneHotEncoderConfig
    | NpConcatenateConfig
  )[];
};
export type TorchModelSpec = {
  name: string;
  framework?: 'torch';
  spec: TorchModelSchema;
  dataset: TorchDatasetConfig;
};
export type DatasetConfig = {
  name: string;
  targetColumns: ColumnConfig[];
  featureColumns: ColumnConfig[];
  featurizers?: (
    | FpVecFilteredTransformerConfig
    | FleetmoleculefeaturizerLayerConfig
    | FleetmoleculefeaturizerLayerConfig2
    | FleetintegerfeaturizerLayerConfig
    | FleetdnasequencefeaturizerLayerConfig
    | FleetrnasequencefeaturizerLayerConfig
    | FleetproteinsequencefeaturizerLayerConfig
  )[];
  transforms?: (
    | StandardScalerConfig
    | LabelEncoderConfig
    | OneHotEncoderConfig
    | NpConcatenateConfig
  )[];
};
export type KNeighborsRegressorConstructorArgs = {
  n_neighbors?: number;
  algorithm?: 'kd_tree';
};
export type KNeighborsRegressorConfig = {
  type?: 'sklearn.neighbors.KNeighborsRegressor';
  constructorArgs?: KNeighborsRegressorConstructorArgs;
  fitArgs: {
    [key: string]: string;
  };
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
};
export type RandomForestRegressorConstructorArgs = {
  n_estimators?: number;
  max_depth?: number;
  min_samples_split?: number | number;
  min_samples_leaf?: number | number;
  min_weight_fraction_leaf?: number;
  max_features?: number | ('sqrt' | 'log2');
  max_leaf_nodes?: number;
  min_impurity_decrease?: number;
  bootstrap?: boolean;
  oob_score?: boolean;
  n_jobs?: number;
  ccp_alpha?: number;
  max_samples?: number | number;
};
export type RandomForestRegressorConfig = {
  type?: 'sklearn.ensemble.RandomForestRegressor';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: RandomForestRegressorConstructorArgs;
  fitArgs: {
    [key: string]: string;
  };
};
export type ExtraTreesRegressorConstructorArgs = {
  n_estimators?: number;
  criterion?: 'squared_error' | 'absolute_error' | 'friedman_mse' | 'poisson';
  max_depth?: number;
  min_samples_split?: number | number;
  min_samples_leaf?: number | number;
  min_weight_fraction_leaf?: number;
  max_features?: number | ('sqrt' | 'log2');
  max_leaf_nodes?: number;
  min_impurity_decrease?: number;
  bootstrap?: boolean;
  oob_score?: boolean;
  n_jobs?: number;
  random_state?: number;
  verbose?: number;
  warm_start?: boolean;
  ccp_alpha?: number;
  max_samples?: number | number;
};
export type ExtraTreesRegressorConfig = {
  type?: 'sklearn.ensemble.ExtraTreesRegressor';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: ExtraTreesRegressorConstructorArgs;
  fitArgs: {
    [key: string]: string;
  };
};
export type ExtraTreesClassifierConstructorArgs = {
  n_estimators?: number;
  criterion?: 'gini' | 'entropy' | 'log_loss';
  max_depth?: number;
  min_samples_split?: number | number;
  min_samples_leaf?: number | number;
  min_weight_fraction_leaf?: number;
  max_features?: number | number | ('sqrt' | 'log2');
  max_leaf_nodes?: number;
  min_impurity_decrease?: number;
  bootstrap?: boolean;
  oob_score?: boolean;
  n_jobs?: number;
  random_state?: number;
  verbose?: number;
  warm_start?: boolean;
  class_weight?: ('balanced' | 'balanced_subsample') | object | object[];
  ccp_alpha?: number;
  max_samples?: number | number;
};
export type ExtraTreesClassifierConfig = {
  type?: 'sklearn.ensemble.ExtraTreesClassifier';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: ExtraTreesClassifierConstructorArgs;
  fitArgs: {
    [key: string]: string;
  };
};
export type KnearestNeighborsClassifierConstructorArgs = {
  n_neighbors?: number;
  weights?: 'uniform' | 'distance';
  algorithm?: 'auto' | 'ball_tree' | 'kd_tree' | 'brute';
  leaf_size?: number;
  p?: number;
  metric?: string;
  metric_params?: {
    [key: string]: string | number | number;
  };
  n_jobs?: number;
};
export type KnearestNeighborsClassifierConfig = {
  type?: 'sklearn.neighbors.KNeighborsClassifier';
  constructorArgs?: KnearestNeighborsClassifierConstructorArgs;
  fitArgs: {
    [key: string]: string;
  };
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
};
export type RandomForestClassifierConstructorArgs = {
  n_estimators?: number;
  criterion?: 'gini' | 'entropy' | 'log_loss';
  max_depth?: number;
  min_samples_split?: number | number;
  min_samples_leaf?: number | number;
  min_weight_fraction_leaf?: number;
  max_features?: number | number | ('sqrt' | 'log2');
  max_leaf_nodes?: number;
  min_impurity_decrease?: number;
  bootstrap?: boolean;
  oob_score?: boolean;
  n_jobs?: number;
  random_state?: number;
  verbose?: number;
  warm_start?: boolean;
  class_weight?: ('balanced' | 'balanced_subsample') | object | object[];
  ccp_alpha?: number;
  max_samples?: number | number;
};
export type RandomForestClassifierConfig = {
  type?: 'sklearn.ensemble.RandomForestClassifier';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: RandomForestClassifierConstructorArgs;
  fitArgs: {
    [key: string]: string;
  };
};
export type SklearnModelSchema = {
  model:
    | KNeighborsRegressorConfig
    | RandomForestRegressorConfig
    | ExtraTreesRegressorConfig
    | ExtraTreesClassifierConfig
    | KnearestNeighborsClassifierConfig
    | RandomForestClassifierConfig;
};
export type SklearnModelSpec = {
  framework?: 'sklearn';
  name: string;
  dataset: DatasetConfig;
  spec: SklearnModelSchema;
};
export type ModelVersion = {
  id: number;
  modelId: number;
  name: string;
  description?: string;
  mlflowVersion?: string;
  mlflowModelName: string;
  config:
    | ({
        framework: 'torch';
      } & TorchModelSpec)
    | ({
        framework: 'sklearn';
      } & SklearnModelSpec);
  createdAt: string;
  updatedAt: string;
};
export type User = {
  email?: string;
  isActive?: boolean;
  isSuperuser?: boolean;
  fullName?: string;
  id?: number;
};
export type Experiment = {
  experimentName?: string;
  modelVersionId: number;
  modelVersion: ModelVersion;
  createdAt: string;
  updatedAt: string;
  createdById: number;
  id: number;
  mlflowId?: string;
  stage: 'NOT RUNNING' | 'RUNNING' | 'SUCCESS' | 'ERROR';
  createdBy?: User;
  hyperparams?: {
    [key: string]: number;
  };
  epochs?: number;
  trainMetrics?: {
    [key: string]: number;
  };
  valMetrics?: {
    [key: string]: number;
  };
  testMetrics?: {
    [key: string]: number;
  };
  history?: {
    [key: string]: number[];
  };
  stackTrace?: string;
};
export type PaginatedExperiment = {
  data: Experiment[];
  total: number;
};
export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
};
export type HttpValidationError = {
  detail?: ValidationError[];
};
export type MonitoringConfig = {
  metricKey: string;
  mode: string;
};
export type AdamParams = {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
};
export type AdamOptimizer = {
  classPath?: 'torch.optim.Adam';
  params?: AdamParams;
};
export type SgdParams = {
  lr?: number;
  momentum?: number;
};
export type SgdOptimizer = {
  classPath?: 'torch.optim.SGD';
  params: SgdParams;
};
export type EarlyStoppingConfig = {
  metricKey: string;
  mode: string;
  minDelta?: number;
  patience?: number;
  checkFinite?: boolean;
};
export type TorchTrainingConfig = {
  epochs: number;
  batchSize?: number;
  checkpointConfig?: MonitoringConfig;
  optimizer?:
    | ({
        classPath: 'torch.optim.Adam';
      } & AdamOptimizer)
    | ({
        classPath: 'torch.optim.SGD';
      } & SgdOptimizer);
  earlyStoppingConfig?: EarlyStoppingConfig;
};
export type BaseTrainingRequest = {
  name: string;
  modelVersionId: number;
  framework: string;
  config: TorchTrainingConfig;
};
export type RunningHistory = {
  experimentId: number;
  userId: number;
  runningHistory: {
    [key: string]: number[];
  };
};
export type MonitorableMetric = {
  key: string;
  label: string;
  texLabel?: string;
  type: 'regressor' | 'classification';
};
export type InputDescription = {
  paramType: 'float' | 'float?';
  default?: any;
  label: string;
};
export type AdamParamsSchema = {
  classPath?: 'torch.optim.Adam';
  lr?: InputDescription;
  beta1?: InputDescription;
  beta2?: InputDescription;
  eps?: InputDescription;
};
export type SgdParamsSchema = {
  classPath?: 'torch.optim.SGD';
  lr?: InputDescription;
  momentum?: InputDescription;
};
export const {
  useGetExperimentsQuery,
  useLazyGetExperimentsQuery,
  usePostExperimentsMutation,
  useGetExperimentsRunningHistoryQuery,
  useLazyGetExperimentsRunningHistoryQuery,
  useGetExperimentsMetricsQuery,
  useLazyGetExperimentsMetricsQuery,
  useGetTrainingExperimentOptimizersQuery,
  useLazyGetTrainingExperimentOptimizersQuery,
  useGetExperimentsMetricsForModelVersionQuery,
  useLazyGetExperimentsMetricsForModelVersionQuery,
} = injectedRtkApi;
