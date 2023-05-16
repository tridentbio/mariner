import {
  TorchModelSpec,
  ColumnConfig as APIColumnConfig,
  TargetConfig as APITargetConfig,
  ColumnsDescription,
  TorchlinearLayerConfig,
  TorchreluLayerConfig,
  TorchsigmoidLayerConfig,
  TorchgeometricgcnconvLayerConfig,
  GetModelOptionsApiResponse,
  NumericalDataType,
  TorchembeddingLayerConfig,
  TorchtransformerencoderlayerLayerConfig,
  FleetconcatLayerConfig,
  FleetonehotLayerConfig,
  FleetglobalpoolingLayerConfig,
  FleetaddpoolingLayerConfig,
  FleetmoleculefeaturizerLayerConfig,
  FleetdnasequencefeaturizerLayerConfig,
  FleetrnasequencefeaturizerLayerConfig,
  FleetproteinsequencefeaturizerLayerConfig,
  FleetintegerfeaturizerLayerConfig,
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
export type LayersType = ArrayElement<TorchModelSpec['spec']['layers']>;

export type FeaturizersType = ArrayElement<
  TorchModelSpec['dataset']['featurizers']
>;
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
  columnType?: TargetConfig['columnType'];
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
export type Concat = FleetconcatLayerConfig;
export type OneHot = FleetonehotLayerConfig;
export type GlobalPooling = FleetglobalpoolingLayerConfig;
export type AddPooling = FleetaddpoolingLayerConfig;
export type MolFeaturizer = FleetmoleculefeaturizerLayerConfig;
export type DNAFeaturizer = FleetdnasequencefeaturizerLayerConfig;
export type RNAFeaturizer = FleetrnasequencefeaturizerLayerConfig;
export type ProteinFeaturizer = FleetproteinsequencefeaturizerLayerConfig;
export type IntegerFeaturizer = FleetintegerfeaturizerLayerConfig;

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
  featurizers: FeaturizersType[];
}
export interface ModelSchema extends TorchModelSpec {
  dataset: DatasetWithForwards;
}
