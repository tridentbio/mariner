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
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as enhancedApi };
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
  /** status 200 Successful Response */ Deployment;
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
  /** status 200 Successful Response */ Deployment;
export type GetPublicDeploymentApiArg = {
  token: string;
};
export type PostMakePredictionDeploymentApiResponse =
  /** status 200 Successful Response */ object;
export type PostMakePredictionDeploymentApiArg = {
  deploymentId: number;
  body: object;
};
export type DeploymentStatus = 'stopped' | 'active' | 'idle' | 'starting';
export type ShareStrategy = 'public' | 'private';
export type RateLimitUnit = 'minute' | 'hour' | 'day' | 'month';
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
} = injectedRtkApi;
