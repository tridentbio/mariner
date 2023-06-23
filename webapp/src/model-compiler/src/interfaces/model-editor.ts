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
  [K in LayerFeaturizerType as K['type']]: K;
};
export type ComponentConfigTypeMap<
  T extends { type: LayerFeaturizerType['type'] }
> = {
  [K in T as K['type']]: K;
};
export type RequiredType<T extends { type?: K }, K extends string> = T & {
  type: K;
};
export type ComponentConfigClassPathMap<
  T extends { classPath: LayerFeaturizerType['type'] }
> = {
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

export type Linear = TorchlinearLayerConfig & { type: 'torch.nn.Linear' };
export type Relu = TorchreluLayerConfig & { type: 'torch.nn.ReLU' };
export type Sigmoid = TorchsigmoidLayerConfig & { type: 'torch.nn.Sigmoid' };
export type GcnConv = TorchgeometricgcnconvLayerConfig & {
  type: 'torch_geometric.nn.GCNConv';
};
export type Embedding = TorchembeddingLayerConfig & {
  type: 'torch.nn.Embedding';
};
export type TransformerEncoderLayer =
  TorchtransformerencoderlayerLayerConfig & {
    type: 'torch.nn.TransformerEncoderLayer';
  };
export type Concat = FleetconcatLayerConfig & {
  type: 'fleet.model_builder.layers.Concat';
};
export type OneHot = FleetonehotLayerConfig & {
  type: 'fleet.model_builder.layers.OneHot';
};
export type GlobalPooling = FleetglobalpoolingLayerConfig & {
  type: 'fleet.model_builder.layers.GlobalPooling';
};
export type AddPooling = FleetaddpoolingLayerConfig & {
  type: 'fleet.model_builder.layers.AddPooling';
};
export type MolFeaturizer = FleetmoleculefeaturizerLayerConfig & {
  type: 'fleet.model_builder.featurizers.MoleculeFeaturizer';
};
export type DNAFeaturizer = FleetdnasequencefeaturizerLayerConfig & {
  type: 'fleet.model_builder.featurizers.DNASequenceFeaturizer';
};
export type RNAFeaturizer = FleetrnasequencefeaturizerLayerConfig & {
  type: 'fleet.model_builder.featurizers.RNASequenceFeaturizer';
};
export type ProteinFeaturizer = FleetproteinsequencefeaturizerLayerConfig & {
  type: 'fleet.model_builder.featurizers.ProteinSequenceFeaturizer';
};
export type IntegerFeaturizer = FleetintegerfeaturizerLayerConfig & {
  type: 'fleet.model_builder.featurizers.IntegerFeaturizer';
};

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
