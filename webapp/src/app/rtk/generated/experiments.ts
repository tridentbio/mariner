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
          body: queryArg.trainingRequest,
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
  trainingRequest: TrainingRequest;
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
export type TargetConfig = {
  name: string;
  dataType:
    | QuantityDataType
    | NumericalDataType
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
    | NumericalDataType
    | StringDataType
    | SmileDataType
    | CategoricalDataType
    | DnaDataType
    | RnaDataType
    | ProteinDataType;
};
export type DatasetConfig = {
  name: string;
  targetColumns: TargetConfig[];
  featureColumns: ColumnConfig[];
};
export type ModelbuilderonehotForwardArgsReferences = {
  x1: string;
};
export type ModelbuilderonehotLayerConfig = {
  type?: 'model_builder.layers.OneHot';
  name: string;
  forwardArgs: ModelbuilderonehotForwardArgsReferences;
};
export type ModelbuilderglobalpoolingConstructorArgs = {
  aggr: string;
};
export type ModelbuilderglobalpoolingForwardArgsReferences = {
  x: string;
  batch?: string;
  size?: string;
};
export type ModelbuilderglobalpoolingLayerConfig = {
  type?: 'model_builder.layers.GlobalPooling';
  name: string;
  constructorArgs: ModelbuilderglobalpoolingConstructorArgs;
  forwardArgs: ModelbuilderglobalpoolingForwardArgsReferences;
};
export type ModelbuilderconcatConstructorArgs = {
  dim?: number;
};
export type ModelbuilderconcatForwardArgsReferences = {
  xs: string[];
};
export type ModelbuilderconcatLayerConfig = {
  type?: 'model_builder.layers.Concat';
  name: string;
  constructorArgs: ModelbuilderconcatConstructorArgs;
  forwardArgs: ModelbuilderconcatForwardArgsReferences;
};
export type ModelbuilderaddpoolingConstructorArgs = {
  dim?: number;
};
export type ModelbuilderaddpoolingForwardArgsReferences = {
  x: string;
};
export type ModelbuilderaddpoolingLayerConfig = {
  type?: 'model_builder.layers.AddPooling';
  name: string;
  constructorArgs: ModelbuilderaddpoolingConstructorArgs;
  forwardArgs: ModelbuilderaddpoolingForwardArgsReferences;
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
};
export type TorchtransformerencoderlayerLayerConfig = {
  type?: 'torch.nn.TransformerEncoderLayer';
  name: string;
  constructorArgs: TorchtransformerencoderlayerConstructorArgs;
  forwardArgs: TorchtransformerencoderlayerForwardArgsReferences;
};
export type ModelbuildermoleculefeaturizerConstructorArgs = {
  allow_unknown: boolean;
  sym_bond_list: boolean;
  per_atom_fragmentation: boolean;
};
export type ModelbuildermoleculefeaturizerForwardArgsReferences = {
  mol: string;
};
export type ModelbuildermoleculefeaturizerLayerConfig = {
  type?: 'model_builder.featurizers.MoleculeFeaturizer';
  name: string;
  constructorArgs: ModelbuildermoleculefeaturizerConstructorArgs;
  forwardArgs: ModelbuildermoleculefeaturizerForwardArgsReferences;
};
export type ModelbuilderintegerfeaturizerForwardArgsReferences = {
  input_: string;
};
export type ModelbuilderintegerfeaturizerLayerConfig = {
  type?: 'model_builder.featurizers.IntegerFeaturizer';
  name: string;
  forwardArgs: ModelbuilderintegerfeaturizerForwardArgsReferences;
};
export type ModelbuilderdnasequencefeaturizerForwardArgsReferences = {
  input_: string;
};
export type ModelbuilderdnasequencefeaturizerLayerConfig = {
  type?: 'model_builder.featurizers.DNASequenceFeaturizer';
  name: string;
  forwardArgs: ModelbuilderdnasequencefeaturizerForwardArgsReferences;
};
export type ModelbuilderrnasequencefeaturizerForwardArgsReferences = {
  input_: string;
};
export type ModelbuilderrnasequencefeaturizerLayerConfig = {
  type?: 'model_builder.featurizers.RNASequenceFeaturizer';
  name: string;
  forwardArgs: ModelbuilderrnasequencefeaturizerForwardArgsReferences;
};
export type ModelbuilderproteinsequencefeaturizerForwardArgsReferences = {
  input_: string;
};
export type ModelbuilderproteinsequencefeaturizerLayerConfig = {
  type?: 'model_builder.featurizers.ProteinSequenceFeaturizer';
  name: string;
  forwardArgs: ModelbuilderproteinsequencefeaturizerForwardArgsReferences;
};
export type ModelSchema = {
  name: string;
  dataset: DatasetConfig;
  layers?: (
    | ({
        type: 'model_builder.layers.OneHot';
      } & ModelbuilderonehotLayerConfig)
    | ({
        type: 'model_builder.layers.GlobalPooling';
      } & ModelbuilderglobalpoolingLayerConfig)
    | ({
        type: 'model_builder.layers.Concat';
      } & ModelbuilderconcatLayerConfig)
    | ({
        type: 'model_builder.layers.AddPooling';
      } & ModelbuilderaddpoolingLayerConfig)
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
  featurizers?: (
    | ({
        type: 'model_builder.featurizers.MoleculeFeaturizer';
      } & ModelbuildermoleculefeaturizerLayerConfig)
    | ({
        type: 'model_builder.featurizers.IntegerFeaturizer';
      } & ModelbuilderintegerfeaturizerLayerConfig)
    | ({
        type: 'model_builder.featurizers.DNASequenceFeaturizer';
      } & ModelbuilderdnasequencefeaturizerLayerConfig)
    | ({
        type: 'model_builder.featurizers.RNASequenceFeaturizer';
      } & ModelbuilderrnasequencefeaturizerLayerConfig)
    | ({
        type: 'model_builder.featurizers.ProteinSequenceFeaturizer';
      } & ModelbuilderproteinsequencefeaturizerLayerConfig)
  )[];
};
export type ModelVersion = {
  id: number;
  modelId: number;
  name: string;
  description?: string;
  mlflowVersion?: string;
  mlflowModelName: string;
  config: ModelSchema;
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
  mlflowId: string;
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
export type TrainingRequest = {
  name: string;
  modelVersionId: number;
  epochs: number;
  batchSize?: number;
  checkpointConfig: MonitoringConfig;
  optimizer:
    | ({
        classPath: 'torch.optim.Adam';
      } & AdamOptimizer)
    | ({
        classPath: 'torch.optim.SGD';
      } & SgdOptimizer);
  earlyStoppingConfig?: EarlyStoppingConfig;
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
