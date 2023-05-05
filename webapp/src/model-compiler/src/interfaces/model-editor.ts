import {
  ModelSchema as APIModelSchema,
  ColumnConfig as APIColumnConfig,
  TargetConfig as APITargetConfig,
  ColumnsDescription,
  TorchlinearLayerConfig,
  TorchreluLayerConfig,
  TorchsigmoidLayerConfig,
  TorchgeometricgcnconvLayerConfig,
  ModelbuilderconcatLayerConfig,
  ModelbuilderonehotLayerConfig,
  ModelbuilderglobalpoolingLayerConfig,
  ModelbuildermoleculefeaturizerLayerConfig,
  GetModelOptionsApiResponse,
  NumericalDataType,
  ModelbuilderdnasequencefeaturizerLayerConfig,
  ModelbuilderrnasequencefeaturizerLayerConfig,
  ModelbuilderproteinsequencefeaturizerLayerConfig,
  ModelbuilderintegerfeaturizerLayerConfig,
  ModelbuilderaddpoolingLayerConfig,
  TorchembeddingLayerConfig,
  TorchtransformerencoderlayerLayerConfig,
} from 'app/rtk/generated/models';

export enum EPythonClasses {
  INT_REQUIRED = "<class 'int'>",
  INT_OPTIONAL = "<class 'int'>?",
  STR_REQUIRED = "<class 'str'>",
  STR_OPTIONAL = "<class 'str'>?",
  BOOL_REQUIRED = "<class 'bool'>",
  BOOL_OPTIONAL = "<class 'bool'>?",
  TORCH_TENSOR_REQUIRED = "<class 'torch.Tensor'>",
  TORCH_TENSOR_OPTIONAL = "<class 'torch.Tensor'>?",
  TORCH_GEOMETRIC_DATA_REQUIRED = "<class 'torch_geometric.data.data.Data'>",
}

type ArrayElement<A> = A extends readonly (infer T)[] ? T : never;

export type ModelOptions = GetModelOptionsApiResponse;
export type LayersType = ArrayElement<ModelSchema['layers']>;

export type FeaturizersType = ArrayElement<ModelSchema['featurizers']>;
export type ComponentType = 'layer' | 'featurizer' | 'input' | 'output';
export type LayerFeaturizerType = LayersType | FeaturizersType;
export type ComponentConfigs = {
  // @ts-ignore
  [K in LayerFeaturizerType as K['type']]: K;
};
export type ComponentConfigTypeMap<
  T extends { type: LayerFeaturizerType['type'] }
> = {
  // @ts-ignore
  [K in T as K['type']]: K;
};
export type RequiredType<T extends { type?: K }, K extends string> = T & {
  type: K;
};
export type ComponentConfigClassPathMap<
  T extends { classPath: LayerFeaturizerType['type'] }
> = {
  // @ts-ignore
  [K in T as K['classPath']]: K;
};
export type ComponentConfigType<T extends LayerFeaturizerType['type']> =
  ComponentConfigs[T];

export type Input = {
  type: 'input';
  name: string;
  dataType: DataType;
};

export type Output = {
  type: 'output';
  name: string;
  dataType: DataType;
  forwardArgs?: { '': string };
  outModule: string;
  columnType: TargetConfig['columnType'];
  lossFn?: TargetConfig['lossFn'];
};

export type NodeType = LayersType | FeaturizersType | Input | Output;

export type DataType = ColumnsDescription['dataType'] | NumericalDataType;

export type Linear = TorchlinearLayerConfig;
export type Relu = TorchreluLayerConfig;
export type Sigmoid = TorchsigmoidLayerConfig;
export type GcnConv = TorchgeometricgcnconvLayerConfig;
export type Embedding = TorchembeddingLayerConfig;
export type TransformerEncoderLayer = TorchtransformerencoderlayerLayerConfig;
export type Concat = ModelbuilderconcatLayerConfig;
export type OneHot = ModelbuilderonehotLayerConfig;
export type GlobalPooling = ModelbuilderglobalpoolingLayerConfig;
export type AddPooling = ModelbuilderaddpoolingLayerConfig;
export type MolFeaturizer = ModelbuildermoleculefeaturizerLayerConfig;
export type DNAFeaturizer = ModelbuilderdnasequencefeaturizerLayerConfig;
export type RNAFeaturizer = ModelbuilderrnasequencefeaturizerLayerConfig;
export type ProteinFeaturizer =
  ModelbuilderproteinsequencefeaturizerLayerConfig;
export type IntegerFeaturizer = ModelbuilderintegerfeaturizerLayerConfig;

type ColumnConfig = APIColumnConfig;
interface ColumnConfigWithForward extends ColumnConfig {
  forwardArgs?: { '': string };
}
type TargetConfig = APITargetConfig;
interface TargetConfigWithForward extends TargetConfig {
  forwardArgs?: { '': string };
}
interface DatasetWithForwards {
  name: string;
  targetColumns: TargetConfigWithForward[];
  featureColumns: ColumnConfigWithForward[];
}
export interface ModelSchema extends Omit<APIModelSchema, 'dataset'> {
  dataset: DatasetWithForwards;
}
