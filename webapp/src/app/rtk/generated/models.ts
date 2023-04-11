import { api } from '../api';
export const addTagTypes = ['models'] as const;
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
          body: queryArg.modelSchema,
        }),
        invalidatesTags: ['models'],
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
  /** status 200 Successful Response */ ForwardCheck;
export type PostModelCheckConfigApiArg = {
  modelSchema: ModelSchema;
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
  outModule?: string;
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
  config: ModelSchema;
};
export type ModelbuilderonehotConstructorArgsSummary = {};
export type ModelbuilderonehotForwardArgsSummary = {
  x1?: string;
};
export type ModelbuilderonehotSummary = {
  type?: 'model_builder.layers.OneHot';
  constructorArgsSummary?: ModelbuilderonehotConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderonehotForwardArgsSummary;
};
export type ModelbuilderglobalpoolingConstructorArgsSummary = {
  aggr?: string;
};
export type ModelbuilderglobalpoolingForwardArgsSummary = {
  x?: string;
  batch?: string;
  size?: string;
};
export type ModelbuilderglobalpoolingSummary = {
  type?: 'model_builder.layers.GlobalPooling';
  constructorArgsSummary?: ModelbuilderglobalpoolingConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderglobalpoolingForwardArgsSummary;
};
export type ModelbuilderconcatConstructorArgsSummary = {
  dim?: string;
};
export type ModelbuilderconcatForwardArgsSummary = {
  xs?: string;
};
export type ModelbuilderconcatSummary = {
  type?: 'model_builder.layers.Concat';
  constructorArgsSummary?: ModelbuilderconcatConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderconcatForwardArgsSummary;
};
export type ModelbuilderaddpoolingConstructorArgsSummary = {
  dim?: string;
};
export type ModelbuilderaddpoolingForwardArgsSummary = {
  x?: string;
};
export type ModelbuilderaddpoolingSummary = {
  type?: 'model_builder.layers.AddPooling';
  constructorArgsSummary?: ModelbuilderaddpoolingConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderaddpoolingForwardArgsSummary;
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
};
export type TorchtransformerencoderlayerSummary = {
  type?: 'torch.nn.TransformerEncoderLayer';
  constructorArgsSummary?: TorchtransformerencoderlayerConstructorArgsSummary;
  forwardArgsSummary?: TorchtransformerencoderlayerForwardArgsSummary;
};
export type ModelbuildermoleculefeaturizerConstructorArgsSummary = {
  allow_unknown?: string;
  sym_bond_list?: string;
  per_atom_fragmentation?: string;
};
export type ModelbuildermoleculefeaturizerForwardArgsSummary = {
  mol?: string;
};
export type ModelbuildermoleculefeaturizerSummary = {
  type?: 'model_builder.featurizers.MoleculeFeaturizer';
  constructorArgsSummary?: ModelbuildermoleculefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: ModelbuildermoleculefeaturizerForwardArgsSummary;
};
export type ModelbuilderintegerfeaturizerConstructorArgsSummary = {};
export type ModelbuilderintegerfeaturizerForwardArgsSummary = {
  input_?: string;
};
export type ModelbuilderintegerfeaturizerSummary = {
  type?: 'model_builder.featurizers.IntegerFeaturizer';
  constructorArgsSummary?: ModelbuilderintegerfeaturizerConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderintegerfeaturizerForwardArgsSummary;
};
export type ModelbuilderdnasequencefeaturizerConstructorArgsSummary = {};
export type ModelbuilderdnasequencefeaturizerForwardArgsSummary = {
  input_?: string;
};
export type ModelbuilderdnasequencefeaturizerSummary = {
  type?: 'model_builder.featurizers.DNASequenceFeaturizer';
  constructorArgsSummary?: ModelbuilderdnasequencefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderdnasequencefeaturizerForwardArgsSummary;
};
export type ModelbuilderrnasequencefeaturizerConstructorArgsSummary = {};
export type ModelbuilderrnasequencefeaturizerForwardArgsSummary = {
  input_?: string;
};
export type ModelbuilderrnasequencefeaturizerSummary = {
  type?: 'model_builder.featurizers.RNASequenceFeaturizer';
  constructorArgsSummary?: ModelbuilderrnasequencefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderrnasequencefeaturizerForwardArgsSummary;
};
export type ModelbuilderproteinsequencefeaturizerConstructorArgsSummary = {};
export type ModelbuilderproteinsequencefeaturizerForwardArgsSummary = {
  input_?: string;
};
export type ModelbuilderproteinsequencefeaturizerSummary = {
  type?: 'model_builder.featurizers.ProteinSequenceFeaturizer';
  constructorArgsSummary?: ModelbuilderproteinsequencefeaturizerConstructorArgsSummary;
  forwardArgsSummary?: ModelbuilderproteinsequencefeaturizerForwardArgsSummary;
};
export type ComponentOption = {
  docsLink?: string;
  docs?: string;
  outputType?: string;
  classPath: string;
  type: 'featurizer' | 'layer';
  component:
    | ModelbuilderonehotSummary
    | ModelbuilderglobalpoolingSummary
    | ModelbuilderconcatSummary
    | ModelbuilderaddpoolingSummary
    | TorchlinearSummary
    | TorchsigmoidSummary
    | TorchreluSummary
    | TorchgeometricgcnconvSummary
    | TorchembeddingSummary
    | TorchtransformerencoderlayerSummary
    | ModelbuildermoleculefeaturizerSummary
    | ModelbuilderintegerfeaturizerSummary
    | ModelbuilderdnasequencefeaturizerSummary
    | ModelbuilderrnasequencefeaturizerSummary
    | ModelbuilderproteinsequencefeaturizerSummary;
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
export type ForwardCheck = {
  stackTrace?: string;
  output?: any;
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
} = injectedRtkApi;
