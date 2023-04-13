import { PaginationQueryParams } from '../paginationQuery';
import { DataType } from './datasets';
import * as modelsApi from 'app/rtk/generated/models';

export type ForwardCheck = modelsApi.ForwardCheck;

export interface ModelCreationError {
  type: 'value_error.missingcomponentargs';
  msg: string;
  log: string[];
  ctx: { [key: string]: unknown };
}

export type ModelConfigDataset = modelsApi.ModelSchema['dataset'];

export type ArrayElement<T> = T extends Array<infer C> ? C : never;
export type Required<T> = T extends undefined ? never : T;
export type Component = Required<
  | ArrayElement<modelsApi.ModelSchema['layers']>
  | ArrayElement<modelsApi.ModelSchema['featurizers']>
>;
export type ModelConfig = modelsApi.ModelSchema;
export type ModelVersion = modelsApi.ModelVersion;
export type ModelVersionType = 'classification' | 'regressor';
export type ModelColumn = modelsApi.ModelFeaturesAndTarget;
export interface ForwardArgs {
  x1?: string;
  x?: string;
  batch?: string;
  size?: string;
  x2?: string;
  input?: string;
  edge_index?: string;
  edge_weight?: string;
  mol?: string;
  xs?: string | string[];
}

export interface ConstructorArgs {
  aggr?: string;
  dim?: number;
  in_features?: number;
  out_features?: number;
  bias?: boolean;
  inplace?: boolean;
  in_channels?: number;
  out_channels?: number;
  improved?: boolean;
  cached?: boolean;
  add_self_loops?: boolean;
  normalize?: boolean;
  allow_unknown?: boolean;
  sym_bond_list?: boolean;
  per_atom_fragmentation?: boolean;
}

export type Model = modelsApi.Model;
export type ArgTypeDict = {
  [key: string]: 'int' | 'string';
};

/**
 *
 * TODO: Fix this type annotation.
 * Should have a fixed set of attributes and arbitrary ones constrained
 * to primitive type strings ('string', 'float', 'int')
 */
export type ComponentConfig = {
  type: string;
} & {
  [key: string]: string | number;
};

export type ComponentAnnotation = {
  docsLink: string;
  docs: string;
  type: 'featurizer' | 'layer';
  classPath: string;
  numInputs: number;
  numOutputs: number;
};

/** Predcition related types **/
export type ModelType = 'numerical' | 'categorical';
export type ModelInputValue = { [key: string]: string[] | number[] };
export type ModelOutputValue = (
  | { [key: string]: string | number | undefined }
  | number
  | number[]
  | number[][]
)[];

export interface ModelQuery extends PaginationQueryParams {
  q?: string;
}

export interface ModelCreateRequest {
  name: string;
  modelDescription: string;
  modelVersionDescription: string;
  config: ModelConfig;
}

export type FeatureTarget = {
  name: string;
  dataType: DataType;
};
