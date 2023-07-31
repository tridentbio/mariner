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

const getType = (python_type: any): TypeIdentifier | undefined => {
  if (typeof python_type !== 'string') return;
  if (python_type.includes('int') || python_type.includes('float'))
    return 'number';
  else if (python_type.includes('bool')) return 'boolean';
  else if (python_type.includes('str')) return 'string';
};

type ScikitType = SklearnModelSchema['model'];

type ComponentType = FeaturizersType | TransformsType | LayersType | ScikitType;

export type TypeIdentifier = 'string' | 'number' | 'boolean';

type ComponentConstructorArgsConfig = {
  [StepKind in ComponentType as StepKind['type']]: StepKind extends {
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

type Option = ArrayElement<GetModelOptionsApiResponse>;
type OptionType = Option['type'];

const PREPROCESSING_OPTIONS: OptionType[] = ['transformer', 'featurizer'];

const LAYER_OPTIONS: OptionType[] = ['layer'];

const SKLEARN_OPTIONS: OptionType[] = ['scikit_class', 'scikit_reg'];

export const toConstructorArgsConfig = (
  option: Option
): ComponentConstructorArgsConfig[keyof ComponentConstructorArgsConfig] => {
  const type = option.classPath;
  const constructorArgs = {};
  if (option.component) {
    for (const [key, value] of Object.entries(
      option.component.constructorArgsSummary
    )) {
      // @ts-ignore
      constructorArgs[key] = {
        type: getType(value),
        // @ts-ignore
        default: option.defaultArgs && option.defaultArgs[key],
        // @ts-ignore
        required: !value.endsWith('?'),
      };
    }
  }
  if (option.defaultArgs) {
    for (const [key, value] of Object.entries(option.defaultArgs)) {
      if (!(key in constructorArgs) && typeof value !== 'object') {
        // @ts-ignore
        constructorArgs[key] = {
          type: typeof value,
          default: value,
        };
      }
    }
  }

  if (Object.keys(constructorArgs).length === 0) {
    return { type } as any;
  }
  if (option.argsOptions) {
    for (const [key, value] of Object.entries(option.argsOptions)) {
      // @ts-ignore
      constructorArgs[key].options = value;
    }
  }
  return {
    type,
    constructorArgs,
  } as any;
};
export default function useModelOptions() {
  const { data, status, error, isLoading } = useGetModelOptionsQuery();

  const sortedData = useMemo(() => {
    const sorted = [...(data || [])];
    sorted.sort((a, b) => {
      if (a.classPath> b.classPath) return 1;
      if (a.classPath< b.classPath) return -1;
      return 0;
    })
    return sorted;
  },[data])

  const getPreprocessingOptions = () => {
    return (sortedData|| []).filter((option) =>
      PREPROCESSING_OPTIONS.includes(option.type)
    );
  };

  const getLayerOptions = () => {
    return (sortedData|| []).filter((option) => LAYER_OPTIONS.includes(option.type));
  };

  const getScikitOptions = (
    returnOnlyTaskType?: 'classification' | 'regression'
  ) => {
    const scikitOptions = (sortedData|| []).filter((option) =>
      SKLEARN_OPTIONS.includes(option.type)
    );
    if (!returnOnlyTaskType) return scikitOptions;
    else if (returnOnlyTaskType === 'classification')
      return (
        scikitOptions &&
        scikitOptions.filter((option) => option.type === 'scikit_class')
      );
    else if (returnOnlyTaskType === 'regression')
      return scikitOptions.filter((option) => option.type === 'scikit_reg');
    throw new Error(`Invalid task type: ${returnOnlyTaskType}`);
  };

  return {
    error,
    getPreprocessingOptions,
    getLayerOptions,
    getScikitOptions,
    options: data,
    isLoading,
  };
}
