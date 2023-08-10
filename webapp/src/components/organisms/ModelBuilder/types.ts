import { ColumnConfig } from '@app/rtk/generated/models';
import { ScikitType, TypeIdentifier } from '@hooks/useModelOptions';
import {
  FeaturizersType,
  TransformsType,
} from '@model-compiler/src/interfaces/model-editor';

export type PreprocessingStep = FeaturizersType | TransformsType | ScikitType;

type PreprocessingStepInputConfig = {
  [StepKind in PreprocessingStep as StepKind['type']]: StepKind extends {
    type: infer F;
    forwardArgs?: any;
    constructorArgs?: infer C;
  }
    ? {
        constructorArgs: {
          [key2 in keyof C]: {
            default: C[key2];
            required?: boolean;
            type: TypeIdentifier;
            options?: string[];
          };
        };
        type: F;
      }
    : never;
};

export type StepValue =
  PreprocessingStepInputConfig[keyof PreprocessingStepInputConfig];

export type GenericPreprocessingStep = {
  type: string;
  constructorArgs?: object;
};
export type SimpleColumnConfig = {
  name: string;
  dataType: ColumnConfig['dataType'];
  featurizers: GenericPreprocessingStep[];
  transforms: GenericPreprocessingStep[];
};
export type DatasetConfigPreprocessing = {
  featureColumns: SimpleColumnConfig[];
  targetColumns: SimpleColumnConfig[];
};
