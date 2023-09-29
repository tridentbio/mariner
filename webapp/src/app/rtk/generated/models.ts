import { api } from '../api';
export const addTagTypes = ['models', 'experiments'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      getModels: build.query<GetModelsApiResponse, GetModelsApiArg>({
        query: (queryArg) => ({
          url: `/api/v1/models/`,
          params: {
            page: queryArg.page,
            perPage: queryArg.perPage,
            q: queryArg.q,
            datasetId: queryArg.datasetId,
          },
        }),
        providesTags: ['models'],
      }),
      createModel: build.mutation<CreateModelApiResponse, CreateModelApiArg>({
        query: (queryArg) => ({
          url: `/api/v1/models/`,
          method: 'POST',
          body: queryArg.modelCreate,
        }),
        invalidatesTags: ['models'],
      }),
      getModelOptions: build.query<
        GetModelOptionsApiResponse,
        GetModelOptionsApiArg
      >({
        query: () => ({ url: `/api/v1/models/options` }),
        providesTags: ['models'],
      }),
      getModelNameSuggestion: build.query<
        GetModelNameSuggestionApiResponse,
        GetModelNameSuggestionApiArg
      >({
        query: () => ({ url: `/api/v1/models/name-suggestion` }),
        providesTags: ['models'],
      }),
      postModelPredict: build.mutation<
        PostModelPredictApiResponse,
        PostModelPredictApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/models/${queryArg.modelVersionId}/predict`,
          method: 'POST',
          body: queryArg.body,
        }),
        invalidatesTags: ['models'],
      }),
      getModelLosses: build.query<
        GetModelLossesApiResponse,
        GetModelLossesApiArg
      >({
        query: () => ({ url: `/api/v1/models/losses` }),
        providesTags: ['models'],
      }),
      getModel: build.query<GetModelApiResponse, GetModelApiArg>({
        query: (queryArg) => ({ url: `/api/v1/models/${queryArg.modelId}` }),
        providesTags: ['models'],
      }),
      deleteModel: build.mutation<DeleteModelApiResponse, DeleteModelApiArg>({
        query: (queryArg) => ({
          url: `/api/v1/models/${queryArg.modelId}`,
          method: 'DELETE',
        }),
        invalidatesTags: ['models'],
      }),
      postModelCheckConfig: build.mutation<
        PostModelCheckConfigApiResponse,
        PostModelCheckConfigApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/models/check-config`,
          method: 'POST',
          body: queryArg.trainingCheckRequest,
        }),
        invalidatesTags: ['models'],
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
export type GetModelsApiResponse =
  /** status 200 Successful Response */ PaginatedModel;
export type GetModelsApiArg = {
  page?: number;
  perPage?: number;
  q?: string;
  datasetId?: number;
};
export type CreateModelApiResponse =
  /** status 200 Successful Response */ Model;
export type CreateModelApiArg = {
  modelCreate: ModelCreate;
};
export type GetModelOptionsApiResponse =
  /** status 200 Successful Response */ ComponentOption[];
export type GetModelOptionsApiArg = void;
export type GetModelNameSuggestionApiResponse =
  /** status 200 Successful Response */ GetNameSuggestionResponse;
export type GetModelNameSuggestionApiArg = void;
export type PostModelPredictApiResponse =
  /** status 200 Successful Response */ {
    [key: string]: any[];
  };
export type PostModelPredictApiArg = {
  modelVersionId: number;
  body: {
    [key: string]: any[];
  };
};
export type GetModelLossesApiResponse =
  /** status 200 Successful Response */ AllowedLosses;
export type GetModelLossesApiArg = void;
export type GetModelApiResponse = /** status 200 Successful Response */ Model;
export type GetModelApiArg = {
  modelId: number;
};
export type DeleteModelApiResponse =
  /** status 200 Successful Response */ Model;
export type DeleteModelApiArg = {
  modelId: number;
};
export type PostModelCheckConfigApiResponse =
  /** status 200 Successful Response */ TrainingCheckResponse;
export type PostModelCheckConfigApiArg = {
  trainingCheckRequest: TrainingCheckRequest;
};
export type GetExperimentsMetricsForModelVersionApiResponse =
  /** status 200 Successful Response */ Experiment[];
export type GetExperimentsMetricsForModelVersionApiArg = {
  modelVersionId: number;
};
export type User = {
  email?: string;
  isActive?: boolean;
  isSuperuser?: boolean;
  fullName?: string;
  id?: number;
};
export type QuantityDataType = {
  domainKind?: 'numeric';
  unit: string;
};
export type NumericalDataType = {
  domainKind?: 'numeric';
};
export type StringDataType = {
  domainKind?: 'string';
};
export type CategoricalDataType = {
  domainKind?: 'categorical';
  classes: {
    [key: string]: number;
  };
};
export type SmileDataType = {
  domainKind?: 'smiles';
};
export type DnaDataType = {
  domainKind?: 'dna';
  isAmbiguous?: boolean;
};
export type RnaDataType = {
  domainKind?: 'rna';
  isAmbiguous?: boolean;
};
export type ProteinDataType = {
  domainKind?: 'protein';
};
export type ColumnsDescription = {
  dataType:
    | QuantityDataType
    | NumericalDataType
    | StringDataType
    | CategoricalDataType
    | SmileDataType
    | DnaDataType
    | RnaDataType
    | ProteinDataType;
  description: string;
  pattern: string;
  datasetId?: number;
};
export type Dataset = {
  id: number;
  name: string;
  description: string;
  rows?: number;
  columns?: number;
  bytes?: number;
  stats?: any;
  dataUrl?: string;
  splitTarget: string;
  splitActual?: string;
  splitType: 'scaffold' | 'random';
  createdAt: string;
  updatedAt: string;
  createdById: number;
  columnsMetadata?: ColumnsDescription[];
  readyStatus?: 'failed' | 'processing' | 'ready';
  errors?: {
    [key: string]: string[] | string;
  };
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
  aggr?: string;
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
export type QuantityDataType2 = {
  domainKind?: 'numeric';
  unit: string;
};
export type NumericDataType = {
  domainKind?: 'numeric';
};
export type StringDataType2 = {
  domainKind?: 'string';
};
export type SmileDataType2 = {
  domainKind?: 'smiles';
};
export type CategoricalDataType2 = {
  domainKind?: 'categorical';
  classes: {
    [key: string]: number;
  };
};
export type DnaDataType2 = {
  domainKind?: 'dna';
};
export type RnaDataType2 = {
  domainKind?: 'rna';
};
export type ProteinDataType2 = {
  domainKind?: 'protein';
};
export type TargetTorchColumnConfig = {
  name: string;
  dataType:
    | QuantityDataType2
    | NumericDataType
    | StringDataType2
    | SmileDataType2
    | CategoricalDataType2
    | DnaDataType2
    | RnaDataType2
    | ProteinDataType2;
  outModule: string;
  lossFn?: string;
  columnType?: 'regression' | 'multiclass' | 'binary';
};
export type ColumnConfig = {
  name: string;
  dataType:
    | QuantityDataType2
    | NumericDataType
    | StringDataType2
    | SmileDataType2
    | CategoricalDataType2
    | DnaDataType2
    | RnaDataType2
    | ProteinDataType2;
};
export type BaseModel = {};
export type OneHotEncoderConfig = {
  type?: 'sklearn.preprocessing.OneHotEncoder';
  constructorArgs?: BaseModel | object;
  name: string;
  forwardArgs:
    | {
        [key: string]: string;
      }
    | string[];
};
export type LabelEncoderConfig = {
  type?: 'sklearn.preprocessing.LabelEncoder';
  constructorArgs?: BaseModel | object;
  name: string;
  forwardArgs:
    | {
        [key: string]: string;
      }
    | string[];
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
export type FleetmoleculefeaturizerLayerConfig2 = {
  type?: 'fleet.model_builder.featurizers.MoleculeFeaturizer';
  name: string;
  constructorArgs: FleetmoleculefeaturizerConstructorArgs;
  forwardArgs: FleetmoleculefeaturizerForwardArgsReferences;
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
export type NpConcatenateConfig = {
  type?: 'fleet.model_builder.transforms.np_concatenate.NpConcatenate';
  constructorArgs?: BaseModel | object;
  name: string;
  forwardArgs:
    | {
        [key: string]: string[];
      }
    | string[];
};
export type TorchDatasetConfig = {
  name: string;
  strategy?: 'forwardArgs';
  targetColumns: TargetTorchColumnConfig[];
  featureColumns: ColumnConfig[];
  featurizers?: (
    | ({
        type: 'sklearn.preprocessing.OneHotEncoder';
      } & OneHotEncoderConfig)
    | ({
        type: 'sklearn.preprocessing.LabelEncoder';
      } & LabelEncoderConfig)
    | ({
        type: 'molfeat.trans.fp.FPVecFilteredTransformer';
      } & FpVecFilteredTransformerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.MoleculeFeaturizer';
      } & FleetmoleculefeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.IntegerFeaturizer';
      } & FleetintegerfeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.DNASequenceFeaturizer';
      } & FleetdnasequencefeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.RNASequenceFeaturizer';
      } & FleetrnasequencefeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.ProteinSequenceFeaturizer';
      } & FleetproteinsequencefeaturizerLayerConfig)
    | ({
        type: 'fleet__preprocessing__FleetmoleculefeaturizerLayerConfig';
      } & FleetmoleculefeaturizerLayerConfig2)
  )[];
  transforms?: (
    | ({
        type: 'sklearn.preprocessing.StandardScaler';
      } & StandardScalerConfig)
    | ({
        type: 'fleet.model_builder.transforms.np_concatenate.NpConcatenate';
      } & NpConcatenateConfig)
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
  strategy?: 'forwardArgs';
  targetColumns: ColumnConfig[];
  featureColumns: ColumnConfig[];
  featurizers?: (
    | ({
        type: 'sklearn.preprocessing.OneHotEncoder';
      } & OneHotEncoderConfig)
    | ({
        type: 'sklearn.preprocessing.LabelEncoder';
      } & LabelEncoderConfig)
    | ({
        type: 'molfeat.trans.fp.FPVecFilteredTransformer';
      } & FpVecFilteredTransformerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.MoleculeFeaturizer';
      } & FleetmoleculefeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.IntegerFeaturizer';
      } & FleetintegerfeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.DNASequenceFeaturizer';
      } & FleetdnasequencefeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.RNASequenceFeaturizer';
      } & FleetrnasequencefeaturizerLayerConfig)
    | ({
        type: 'fleet.model_builder.featurizers.ProteinSequenceFeaturizer';
      } & FleetproteinsequencefeaturizerLayerConfig)
    | ({
        type: 'fleet__preprocessing__FleetmoleculefeaturizerLayerConfig';
      } & FleetmoleculefeaturizerLayerConfig2)
  )[];
  transforms?: (
    | ({
        type: 'sklearn.preprocessing.StandardScaler';
      } & StandardScalerConfig)
    | ({
        type: 'fleet.model_builder.transforms.np_concatenate.NpConcatenate';
      } & NpConcatenateConfig)
  )[];
};
export type CreateFromType = {
  type: string;
  constructorArgs?: BaseModel | object;
};
export type ColumnConfigWithPreprocessing = {
  name: string;
  dataType:
    | NumericDataType
    | CategoricalDataType2
    | QuantityDataType2
    | SmileDataType2
    | DnaDataType2
    | RnaDataType2
    | ProteinDataType2;
  transforms?: CreateFromType[];
  featurizers?: CreateFromType[];
};
export type DatasetConfigWithPreprocessing = {
  name: string;
  strategy?: 'pipeline';
  targetColumns: ColumnConfigWithPreprocessing[];
  featureColumns: ColumnConfigWithPreprocessing[];
};
export type KNeighborsRegressorConstructorArgs = {
  n_neighbors?: number;
  algorithm?: 'kd_tree';
};
export type KNeighborsRegressorConfig = {
  type?: 'sklearn.neighbors.KNeighborsRegressor';
  constructorArgs?: KNeighborsRegressorConstructorArgs;
  fitArgs?: {
    [key: string]: string;
  };
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
};
export type RandomForestRegressorConstructorArgs = {
  n_estimators?: number;
  criterion?: 'squared_error' | 'absolute_error' | 'friedman_mse' | 'poisson';
};
export type RandomForestRegressorConfig = {
  type?: 'sklearn.ensemble.RandomForestRegressor';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: RandomForestRegressorConstructorArgs;
  fitArgs?: {
    [key: string]: string;
  };
};
export type ExtraTreesRegressorConstructorArgs = {
  n_estimators?: number;
  criterion?: 'squared_error' | 'absolute_error' | 'friedman_mse' | 'poisson';
};
export type ExtraTreesRegressorConfig = {
  type?: 'sklearn.ensemble.ExtraTreesRegressor';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: ExtraTreesRegressorConstructorArgs;
  fitArgs?: {
    [key: string]: string;
  };
};
export type ExtraTreesClassifierConstructorArgs = {
  n_estimators?: number;
  criterion?: 'gini' | 'entropy' | 'log_loss';
};
export type ExtraTreesClassifierConfig = {
  type?: 'sklearn.ensemble.ExtraTreesClassifier';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: ExtraTreesClassifierConstructorArgs;
  fitArgs?: {
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
  fitArgs?: {
    [key: string]: string;
  };
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
};
export type RandomForestClassifierConstructorArgs = {
  n_estimators?: number;
  criterion?: 'gini' | 'entropy' | 'log_loss';
};
export type RandomForestClassifierConfig = {
  type?: 'sklearn.ensemble.RandomForestClassifier';
  taskType?: ('regressor' | 'multiclass' | 'multilabel')[];
  constructorArgs?: RandomForestClassifierConstructorArgs;
  fitArgs?: {
    [key: string]: string;
  };
};
export type SklearnModelSchema = {
  model:
    | ({
        type: 'sklearn.neighbors.KNeighborsRegressor';
      } & KNeighborsRegressorConfig)
    | ({
        type: 'sklearn.ensemble.RandomForestRegressor';
      } & RandomForestRegressorConfig)
    | ({
        type: 'sklearn.ensemble.ExtraTreesRegressor';
      } & ExtraTreesRegressorConfig)
    | ({
        type: 'sklearn.ensemble.ExtraTreesClassifier';
      } & ExtraTreesClassifierConfig)
    | ({
        type: 'sklearn.neighbors.KNeighborsClassifier';
      } & KnearestNeighborsClassifierConfig)
    | ({
        type: 'sklearn.ensemble.RandomForestClassifier';
      } & RandomForestClassifierConfig);
};
export type SklearnModelSpec = {
  framework?: 'sklearn';
  name: string;
  dataset:
    | ({
        strategy: 'forwardArgs';
      } & DatasetConfig)
    | ({
        strategy: 'pipeline';
      } & DatasetConfigWithPreprocessing);
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
export type ModelFeaturesAndTarget = {
  modelId?: number;
  columnName: string;
  columnType: 'feature' | 'target';
};
export type Model = {
  id: number;
  name: string;
  mlflowName: string;
  description?: string;
  createdById: number;
  createdBy?: User;
  datasetId?: number;
  dataset?: Dataset;
  versions: ModelVersion[];
  columns: ModelFeaturesAndTarget[];
  createdAt: string;
  updatedAt: string;
};
export type PaginatedModel = {
  data: Model[];
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
export type ModelCreate = {
  name: string;
  modelDescription?: string;
  modelVersionDescription?: string;
  config:
    | ({
        framework: 'torch';
      } & TorchModelSpec)
    | ({
        framework: 'sklearn';
      } & SklearnModelSpec);
};
export type ComponentType =
  | 'transformer'
  | 'featurizer'
  | 'layer'
  | 'scikit_reg'
  | 'scikit_class';
export type ArgumentOptionMetadata = {
  key: string;
  label?: string;
  latex?: string;
};
export type ComponentOption = {
  classPath: string;
  component?: any;
  type: ComponentType;
  argsOptions?: {
    [key: string]: (string | ArgumentOptionMetadata)[];
  };
  docsLink?: string;
  docs?: string;
  outputType?: string;
  defaultArgs?: object;
  //? Declared manually for mocked version
  compatibleWith?: {
    domains?: ColumnConfig['dataType']['domainKind'][];
    framework?: ('torch' | 'sklearn')[];
  };
};
export type GetNameSuggestionResponse = {
  name: string;
};
export type AllowedLosses = {
  regr?: {
    [key: string]: string;
  }[];
  binClass?: {
    [key: string]: string;
  }[];
  mcClass?: {
    [key: string]: string;
  }[];
  typeMap?: object;
};
export type TrainingCheckResponse = {
  stackTrace?: string;
  output?: any;
};
export type TrainingCheckRequest = {
  modelSpec: TorchModelSpec;
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
export const {
  useGetModelsQuery,
  useLazyGetModelsQuery,
  useCreateModelMutation,
  useGetModelOptionsQuery,
  useLazyGetModelOptionsQuery,
  useGetModelNameSuggestionQuery,
  useLazyGetModelNameSuggestionQuery,
  usePostModelPredictMutation,
  useGetModelLossesQuery,
  useLazyGetModelLossesQuery,
  useGetModelQuery,
  useLazyGetModelQuery,
  useDeleteModelMutation,
  usePostModelCheckConfigMutation,
  useGetExperimentsMetricsForModelVersionQuery,
  useLazyGetExperimentsMetricsForModelVersionQuery,
} = injectedRtkApi;
