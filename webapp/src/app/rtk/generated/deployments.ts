import { api } from '../api';
export const addTagTypes = ['deployments'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      getDeployments: build.query<
        GetDeploymentsApiResponse,
        GetDeploymentsApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/`,
          params: {
            page: queryArg.page,
            perPage: queryArg.perPage,
            name: queryArg.name,
            status: queryArg.status,
            shareStrategy: queryArg.shareStrategy,
            createdAfter: queryArg.createdAfter,
            modelVersionId: queryArg.modelVersionId,
            publicMode: queryArg.publicMode,
            accessMode: queryArg.accessMode,
          },
        }),
        providesTags: ['deployments'],
      }),
      createDeployment: build.mutation<
        CreateDeploymentApiResponse,
        CreateDeploymentApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/`,
          method: 'POST',
          body: queryArg.deploymentBase,
        }),
        invalidatesTags: ['deployments'],
      }),
      getDeployment: build.query<GetDeploymentApiResponse, GetDeploymentApiArg>(
        {
          query: (queryArg) => ({
            url: `/api/v1/deployments/${queryArg.deploymentId}`,
          }),
          providesTags: ['deployments'],
        }
      ),
      updateDeployment: build.mutation<
        UpdateDeploymentApiResponse,
        UpdateDeploymentApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/${queryArg.deploymentId}`,
          method: 'PUT',
          body: queryArg.deploymentUpdateInput,
        }),
        invalidatesTags: ['deployments'],
      }),
      deleteDeployment: build.mutation<
        DeleteDeploymentApiResponse,
        DeleteDeploymentApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/${queryArg.deploymentId}`,
          method: 'DELETE',
        }),
        invalidatesTags: ['deployments'],
      }),
      getPublicDeployment: build.query<
        GetPublicDeploymentApiResponse,
        GetPublicDeploymentApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/public/${queryArg.token}`,
        }),
        providesTags: ['deployments'],
      }),
      postMakePredictionDeployment: build.mutation<
        PostMakePredictionDeploymentApiResponse,
        PostMakePredictionDeploymentApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/${queryArg.deploymentId}/predict`,
          method: 'POST',
          body: queryArg.body,
        }),
        invalidatesTags: ['deployments'],
      }),
      postMakePredictionDeploymentPublic: build.mutation<
        PostMakePredictionDeploymentPublicApiResponse,
        PostMakePredictionDeploymentPublicApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/${queryArg.deploymentId}/predict-public`,
          method: 'POST',
          body: queryArg.body,
        }),
        invalidatesTags: ['deployments'],
      }),
      handleDeploymentManager: build.mutation<
        HandleDeploymentManagerApiResponse,
        HandleDeploymentManagerApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/deployments/deployment-manager`,
          method: 'POST',
          body: queryArg.deploymentManagerComunication,
          headers: { authorization: queryArg.authorization },
        }),
        invalidatesTags: ['deployments'],
      }),
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as generatedDeploymentsApi };
export type GetDeploymentsApiResponse =
  /** status 200 Successful Response */ PaginatedDeployment;
export type GetDeploymentsApiArg = {
  page?: number;
  perPage?: number;
  name?: string;
  status?: DeploymentStatus;
  shareStrategy?: ShareStrategy;
  createdAfter?: string;
  modelVersionId?: number;
  publicMode?: 'include' | 'exclude' | 'only';
  accessMode?: 'unset' | 'owned' | 'shared';
};
export type CreateDeploymentApiResponse =
  /** status 200 Successful Response */ Deployment;
export type CreateDeploymentApiArg = {
  deploymentBase: DeploymentBase;
};
export type GetDeploymentApiResponse =
  /** status 200 Successful Response */ DeploymentWithTrainingData;
export type GetDeploymentApiArg = {
  deploymentId: number;
};
export type UpdateDeploymentApiResponse =
  /** status 200 Successful Response */ Deployment;
export type UpdateDeploymentApiArg = {
  deploymentId: number;
  deploymentUpdateInput: DeploymentUpdateInput;
};
export type DeleteDeploymentApiResponse =
  /** status 200 Successful Response */ Deployment;
export type DeleteDeploymentApiArg = {
  deploymentId: number;
};
export type GetPublicDeploymentApiResponse =
  /** status 200 Successful Response */ DeploymentWithTrainingData;
export type GetPublicDeploymentApiArg = {
  token: string;
};
export type PostMakePredictionDeploymentApiResponse =
  /** status 200 Successful Response */ object;
export type PostMakePredictionDeploymentApiArg = {
  deploymentId: number;
  body: object;
};
export type PostMakePredictionDeploymentPublicApiResponse =
  /** status 200 Successful Response */ object;
export type PostMakePredictionDeploymentPublicApiArg = {
  deploymentId: number;
  body: object;
};
export type HandleDeploymentManagerApiResponse =
  /** status 200 Successful Response */ Deployment | Deployment[];
export type HandleDeploymentManagerApiArg = {
  authorization?: string;
  deploymentManagerComunication: DeploymentManagerComunication;
};
export type DeploymentStatus = 'stopped' | 'active' | 'idle' | 'starting';
export type ShareStrategy = 'public' | 'private';
export type RateLimitUnit = 'minute' | 'hour' | 'day' | 'month';
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
    | CategoricalDataType
    | QuantityDataType
    | SmileDataType
    | DnaDataType
    | RnaDataType
    | ProteinDataType;
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
  checkStatus?: 'OK' | 'FAILED' | 'RUNNING';
  checkStackTrace?: string;
};
export type User = {
  id: number;
  email: string;
  fullName?: string;
};
export type Deployment = {
  name: string;
  readme?: string;
  shareUrl?: string;
  status?: DeploymentStatus;
  modelVersionId: number;
  shareStrategy?: ShareStrategy;
  usersIdAllowed?: number[];
  organizationsAllowed?: string[];
  showTrainingData?: boolean;
  predictionRateLimitValue: number;
  predictionRateLimitUnit?: RateLimitUnit;
  deletedAt?: string;
  id: number;
  createdById: number;
  modelVersion?: ModelVersion;
  usersAllowed?: User[];
  createdAt: string;
  updatedAt: string;
};
export type PaginatedDeployment = {
  data: Deployment[];
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
export type DeploymentBase = {
  name: string;
  readme?: string;
  shareUrl?: string;
  status?: DeploymentStatus;
  modelVersionId: number;
  shareStrategy?: ShareStrategy;
  usersIdAllowed?: number[];
  organizationsAllowed?: string[];
  showTrainingData?: boolean;
  predictionRateLimitValue: number;
  predictionRateLimitUnit?: RateLimitUnit;
  deletedAt?: string;
};
export type DatasetSummary = {
  train: object;
  val: object;
  test: object;
  full: object;
};
export type DeploymentWithTrainingData = {
  name: string;
  readme?: string;
  shareUrl?: string;
  status?: DeploymentStatus;
  modelVersionId: number;
  shareStrategy?: ShareStrategy;
  usersIdAllowed?: number[];
  organizationsAllowed?: string[];
  showTrainingData?: boolean;
  predictionRateLimitValue: number;
  predictionRateLimitUnit?: RateLimitUnit;
  deletedAt?: string;
  id: number;
  createdById: number;
  modelVersion?: ModelVersion;
  usersAllowed?: User[];
  createdAt: string;
  updatedAt: string;
  datasetSummary?: DatasetSummary;
};
export type DeploymentUpdateInput = {
  name?: string;
  readme?: string;
  status?: DeploymentStatus;
  shareStrategy?: ShareStrategy;
  usersIdAllowed?: number[];
  organizationsAllowed?: string[];
  showTrainingData?: boolean;
  predictionRateLimitValue?: number;
  predictionRateLimitUnit?: RateLimitUnit;
  modelVersionId?: number;
};
export type DeploymentManagerComunication = {
  deploymentId: number;
  status: DeploymentStatus;
};
export const {
  useGetDeploymentsQuery,
  useLazyGetDeploymentsQuery,
  useCreateDeploymentMutation,
  useGetDeploymentQuery,
  useLazyGetDeploymentQuery,
  useUpdateDeploymentMutation,
  useDeleteDeploymentMutation,
  useGetPublicDeploymentQuery,
  useLazyGetPublicDeploymentQuery,
  usePostMakePredictionDeploymentMutation,
  usePostMakePredictionDeploymentPublicMutation,
  useHandleDeploymentManagerMutation,
} = injectedRtkApi;
