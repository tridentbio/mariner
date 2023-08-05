import { SimpleColumnConfig } from '@components/organisms/ModelBuilder/types';
import {
  TorchModelSpec,
  TargetTorchColumnConfig,
  ColumnConfig,
  ColumnsDescription,
  TorchlinearLayerConfig,
  TorchreluLayerConfig,
  TorchsigmoidLayerConfig,
  TorchgeometricgcnconvLayerConfig,
  GetModelOptionsApiResponse,
  NumericDataType,
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
  SklearnModelSpec,
} from 'app/rtk/generated/models';

export type APITargetConfig = TargetTorchColumnConfig;
export type APIColumnConfig = ColumnConfig;
export type APISimpleColumnConfig = SimpleColumnConfig;

export type FleetModelSpec = TorchModelSpec | SklearnModelSpec;

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

export type TransformsType = ArrayElement<
  TorchModelSpec['dataset']['transforms']
>;
export type ComponentType =
  | 'layer'
  | 'featurizer'
  | 'transformer'
  | 'input'
  | 'output';
export type LayerFeaturizerType = LayersType | FeaturizersType | TransformsType;
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
  columnType?: APITargetConfig['columnType'];
  lossFn?: APITargetConfig['lossFn'];
};

export type NodeType = LayersType | FeaturizersType | Input | Output;

export type DataType = ColumnsDescription['dataType'] | NumericDataType;

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

interface ColumnConfigWithForward extends APIColumnConfig {
  forwardArgs?: { '': string };
}
interface TargetConfigWithForward extends APITargetConfig {
  forwardArgs?: { '': string };
}
interface SimpleColumnConfigWithForward extends SimpleColumnConfig {
  forwardArgs?: { '': string };
}
interface DatasetWithForwards<TargetColumnConfig, FeatureColumnConfig> {
  name: string;
  targetColumns: TargetColumnConfig[];
  featureColumns: FeatureColumnConfig[];
  featurizers: FeaturizersType[];
  transforms: TransformsType[];
}

export interface TorchModel extends TorchModelSpec {
  dataset: DatasetWithForwards<
    TargetConfigWithForward,
    ColumnConfigWithForward
  >;
}

export interface SkLearnModel extends SklearnModelSpec {
  dataset: DatasetWithForwards<
    SimpleColumnConfigWithForward,
    SimpleColumnConfigWithForward
  >;
}

export type ModelSchema<Model extends 'torch' | 'sklearn' = 'torch'> =
  Model extends 'torch' ? TorchModel : SkLearnModel;
