import {
  GetModelOptionsApiResponse,
  SklearnModelSchema,
  useGetModelOptionsQuery,
} from '@app/rtk/generated/models';
import {
  FeaturizersType,
  LayersType,
  TransformsType,
} from '@model-compiler/src/interfaces/model-editor';
import { ArrayElement } from '@utils';

type ScikitType = SklearnModelSchema['model'];

type OptionType = FeaturizersType | TransformsType | LayersType | ScikitType;

type TypeIdentifier = 'string' | 'number' | 'bool';

type PreprocessingStepInputConfig = {
  [StepKind in OptionType as StepKind['type']]: StepKind extends {
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

const PREPROCESSING_OPTIONS: ArrayElement<GetModelOptionsApiResponse>['type'][] =
  ['transformer', 'featurizer'];

const LAYER_OPTIONS: ArrayElement<GetModelOptionsApiResponse>['type'][] = [
  'layer',
];

const SKLEARN_OPTIONS: ArrayElement<GetModelOptionsApiResponse>['type'][] = [
  'scikit_class',
  'scikit_reg',
];
export default function useModelOptions() {
  const { data, status, error } = useGetModelOptionsQuery();

  const getPreprocessingOptions = () => {
    return (
      data &&
      data.filter((option) => PREPROCESSING_OPTIONS.includes(option.type))
    );
  };

  const getLayerOptions = () => {
    return data && data.filter((option) => LAYER_OPTIONS.includes(option.type));
  };

  const getScikitOptions = (
    returnOnlyTaskType?: 'classification' | 'regression'
  ) => {
    const scikitOptions =
      data && data.filter((option) => SKLEARN_OPTIONS.includes(option.type));
    if (!returnOnlyTaskType) return scikitOptions;
    else if (returnOnlyTaskType === 'classification')
      return (
        scikitOptions &&
        scikitOptions.filter((option) => option.type === 'scikit_class')
      );
    else if (returnOnlyTaskType === 'regression')
      return (
        scikitOptions &&
        scikitOptions.filter((option) => option.type === 'scikit_reg')
      );
  };

  return {
    getPreprocessingOptions,
    getLayerOptions,
    getScikitOptions,
    error,
  };
}
