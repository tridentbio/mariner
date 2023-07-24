import { ColumnConfig } from '@app/rtk/generated/models';

export type PreprocessingConfig = {
  name: string;
  constructorArgs: Record<string, any>;
  fowardArgs: Record<string, null>;
  type: string;
};

export type Transforms = PreprocessingConfig[];

export type Featurizers = PreprocessingConfig[];

export type FormColumns = Record<
  'feature' | 'target',
  {
    col: ColumnConfig;
    transforms: Transforms;
    featurizers: Featurizers;
  }[]
>;
