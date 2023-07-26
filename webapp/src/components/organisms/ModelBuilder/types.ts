import {
  FeaturizersType,
  TransformsType,
} from '@model-compiler/src/interfaces/model-editor';

type PreprocessingStep = FeaturizersType | TransformsType;

export type TypeIdentifier = 'string' | 'number' | 'bool';

type PreprocessingStepInputConfig = {
  [StepKind in PreprocessingStep as StepKind['type']]: StepKind extends {
    type: infer F;
    forwardArgs: any;
    constructorArgs?: infer C;
  }
    ? {
        constructorArgs: {
          [key2 in keyof C]: {
            default: C[key2];
            type: TypeIdentifier;
          };
        };
        type: F;
      }
    : never;
};

export type StepValue =
  PreprocessingStepInputConfig[keyof PreprocessingStepInputConfig];
export interface PreprocessingStepSelectProps {
  value?: PreprocessingStep;
  onChange: (step?: PreprocessingStep) => any;
  filterOptions?: (step: PreprocessingStep) => boolean;
  error?: boolean;
  helperText?: string;
  options: StepValue[];
}
