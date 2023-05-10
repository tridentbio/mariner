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
export type QuantityDataType2 = {
  domainKind?: 'numeric';
  unit: string;
};
export type NumericalDataType = {
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
export type TargetConfig = {
  name: string;
  dataType:
    | QuantityDataType2
    | NumericalDataType
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
    | NumericalDataType
    | StringDataType2
    | SmileDataType2
    | CategoricalDataType2
    | DnaDataType2
    | RnaDataType2
    | ProteinDataType2;
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
export type TorchDatasetConfig = {
  name: string;
  targetColumns: TargetConfig[];
  featureColumns: ColumnConfig[];
  featurizers?: (
    | FleetmoleculefeaturizerLayerConfig
    | FleetintegerfeaturizerLayerConfig
    | FleetdnasequencefeaturizerLayerConfig
    | FleetrnasequencefeaturizerLayerConfig
    | FleetproteinsequencefeaturizerLayerConfig
  )[];
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
export type TorchModelSpec = {
  name: string;
  framework?: 'torch';
  dataset: TorchDatasetConfig;
  spec: TorchModelSchema;
};
export type ModelVersion = {
  id: number;
  modelId: number;
  name: string;
  description?: string;
  mlflowVersion?: string;
  mlflowModelName: string;
  config: TorchModelSpec;
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
  config: TorchModelSpec;
};
export type FleetonehotConstructorArgsSummary = {};
export type FleetonehotForwardArgsSummary = {
  x1?: string;
};
export type FleetonehotSummary = {
  type?: 'fleet.model_builder.layers.OneHot';
  constructorArgsSummary?: FleetonehotConstructorArgsSummary;
  forwardArgsSummary?: FleetonehotForwardArgsSummary;
};
export type FleetglobalpoolingConstructorArgsSummary = {
  aggr?: string;
};
export type FleetglobalpoolingForwardArgsSummary = {
  x?: string;
  batch?: string;
  size?: string;
};
export type FleetglobalpoolingSummary = {
  type?: 'fleet.model_builder.layers.GlobalPooling';
  constructorArgsSummary?: FleetglobalpoolingConstructorArgsSummary;
  forwardArgsSummary?: FleetglobalpoolingForwardArgsSummary;
};
export type FleetconcatConstructorArgsSummary = {
  dim?: string;
};
export type FleetconcatForwardArgsSummary = {
  xs?: string;
};
export type FleetconcatSummary = {
  type?: 'fleet.model_builder.layers.Concat';
  constructorArgsSummary?: FleetconcatConstructorArgsSummary;
  forwardArgsSummary?: FleetconcatForwardArgsSummary;
};
export type FleetaddpoolingConstructorArgsSummary = {
  dim?: string;
};
export type FleetaddpoolingForwardArgsSummary = {
  x?: string;
};
export type FleetaddpoolingSummary = {
  type?: 'fleet.model_builder.layers.AddPooling';
  constructorArgsSummary?: FleetaddpoolingConstructorArgsSummary;
  forwardArgsSummary?: FleetaddpoolingForwardArgsSummary;
};
export type TorchlinearConstructorArgsSummary = {
  in_features?: string;
  out_features?: string;
  bias?: string;
};
export type TorchlinearForwardArgsSummary = {
  input?: string;
};
export type TorchlinearSummary = {
  type?: 'torch.nn.Linear';
  constructorArgsSummary?: TorchlinearConstructorArgsSummary;
  forwardArgsSummary?: TorchlinearForwardArgsSummary;
};
export type TorchsigmoidConstructorArgsSummary = {};
export type TorchsigmoidForwardArgsSummary = {
  input?: string;
};
export type TorchsigmoidSummary = {
  type?: 'torch.nn.Sigmoid';
  constructorArgsSummary?: TorchsigmoidConstructorArgsSummary;
  forwardArgsSummary?: TorchsigmoidForwardArgsSummary;
};
export type TorchreluConstructorArgsSummary = {
  inplace?: string;
};
export type TorchreluForwardArgsSummary = {
  input?: string;
};
export type TorchreluSummary = {
  type?: 'torch.nn.ReLU';
  constructorArgsSummary?: TorchreluConstructorArgsSummary;
  forwardArgsSummary?: TorchreluForwardArgsSummary;
};
export type TorchgeometricgcnconvConstructorArgsSummary = {
  in_channels?: string;
  out_channels?: string;
  improved?: string;
  cached?: string;
  add_self_loops?: string;
  normalize?: string;
  bias?: string;
};
export type TorchgeometricgcnconvForwardArgsSummary = {
  x?: string;
  edge_index?: string;
  edge_weight?: string;
};
export type TorchgeometricgcnconvSummary = {
  type?: 'torch_geometric.nn.GCNConv';
  constructorArgsSummary?: TorchgeometricgcnconvConstructorArgsSummary;
  forwardArgsSummary?: TorchgeometricgcnconvForwardArgsSummary;
};
export type TorchembeddingConstructorArgsSummary = {
  num_embeddings?: string;
  embedding_dim?: string;
  padding_idx?: string;
  max_norm?: string;
  norm_type?: string;
  scale_grad_by_freq?: string;
  sparse?: string;
};
export type TorchembeddingForwardArgsSummary = {
  input?: string;
};
export type TorchembeddingSummary = {
  type?: 'torch.nn.Embedding';
  constructorArgsSummary?: TorchembeddingConstructorArgsSummary;
  forwardArgsSummary?: TorchembeddingForwardArgsSummary;
};
export type TorchtransformerencoderlayerConstructorArgsSummary = {
  d_model?: string;
  nhead?: string;
  dim_feedforward?: string;
  dropout?: string;
  activation?: string;
  layer_norm_eps?: string;
  batch_first?: string;
  norm_first?: string;
};
export type TorchtransformerencoderlayerForwardArgsSummary = {
  src?: string;
  src_mask?: string;
  src_key_padding_mask?: string;
  is_causal?: string;
};
export type TorchtransformerencoderlayerSummary = {
  type?: 'torch.nn.TransformerEncoderLayer';
  constructorArgsSummary?: TorchtransformerencoderlayerConstructorArgsSummary;
  forwardArgsSummary?: TorchtransformerencoderlayerForwardArgsSummary;
};
export type FleetmoleculefeaturizerConstructorArgsSummary = {
  allow_unknown?: string;
  sym_bond_list?: string;
  per_atom_fragmentation?: string;
};
export type FleetmoleculefeaturizerForwardArgsSummary = {
  mol?: string;
};
export type FleetmoleculefeaturizerSummary = {
  type?: 'fleet.model_builder.featurizers.MoleculeFeaturizer';
  constructorArgsSummary?: FleetmoleculefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: FleetmoleculefeaturizerForwardArgsSummary;
};
export type FleetintegerfeaturizerConstructorArgsSummary = {};
export type FleetintegerfeaturizerForwardArgsSummary = {
  input_?: string;
};
export type FleetintegerfeaturizerSummary = {
  type?: 'fleet.model_builder.featurizers.IntegerFeaturizer';
  constructorArgsSummary?: FleetintegerfeaturizerConstructorArgsSummary;
  forwardArgsSummary?: FleetintegerfeaturizerForwardArgsSummary;
};
export type FleetdnasequencefeaturizerConstructorArgsSummary = {};
export type FleetdnasequencefeaturizerForwardArgsSummary = {
  input_?: string;
};
export type FleetdnasequencefeaturizerSummary = {
  type?: 'fleet.model_builder.featurizers.DNASequenceFeaturizer';
  constructorArgsSummary?: FleetdnasequencefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: FleetdnasequencefeaturizerForwardArgsSummary;
};
export type FleetrnasequencefeaturizerConstructorArgsSummary = {};
export type FleetrnasequencefeaturizerForwardArgsSummary = {
  input_?: string;
};
export type FleetrnasequencefeaturizerSummary = {
  type?: 'fleet.model_builder.featurizers.RNASequenceFeaturizer';
  constructorArgsSummary?: FleetrnasequencefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: FleetrnasequencefeaturizerForwardArgsSummary;
};
export type FleetproteinsequencefeaturizerConstructorArgsSummary = {};
export type FleetproteinsequencefeaturizerForwardArgsSummary = {
  input_?: string;
};
export type FleetproteinsequencefeaturizerSummary = {
  type?: 'fleet.model_builder.featurizers.ProteinSequenceFeaturizer';
  constructorArgsSummary?: FleetproteinsequencefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: FleetproteinsequencefeaturizerForwardArgsSummary;
};
export type ComponentOption = {
  docsLink?: string;
  docs?: string;
  outputType?: string;
  classPath: string;
  type: 'featurizer' | 'layer';
  component:
    | FleetonehotSummary
    | FleetglobalpoolingSummary
    | FleetconcatSummary
    | FleetaddpoolingSummary
    | TorchlinearSummary
    | TorchsigmoidSummary
    | TorchreluSummary
    | TorchgeometricgcnconvSummary
    | TorchembeddingSummary
    | TorchtransformerencoderlayerSummary
    | FleetmoleculefeaturizerSummary
    | FleetintegerfeaturizerSummary
    | FleetdnasequencefeaturizerSummary
    | FleetrnasequencefeaturizerSummary
    | FleetproteinsequencefeaturizerSummary;
  defaultArgs?: object;
  argsOptions?: {
    [key: string]: string[];
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
};
export type TrainingCheckResponse = {
  stackTrace?: string;
  output?: any;
};
export type DatasetConfig = {
  name: string;
  targetColumns: ColumnConfig[];
  featureColumns: ColumnConfig[];
  featurizers?: (
    | FleetmoleculefeaturizerLayerConfig
    | FleetintegerfeaturizerLayerConfig
    | FleetdnasequencefeaturizerLayerConfig
    | FleetrnasequencefeaturizerLayerConfig
    | FleetproteinsequencefeaturizerLayerConfig
  )[];
};
export type BaseFleetModelSpec = {
  name: string;
  framework: string;
  dataset: DatasetConfig;
  spec?: any;
};
export type TrainingCheckRequest = {
  modelSpec: BaseFleetModelSpec;
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
