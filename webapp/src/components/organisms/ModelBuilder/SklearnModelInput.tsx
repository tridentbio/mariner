import {
  ModelCreate,
  SklearnModelSchema,
  SklearnModelSpec,
} from '@app/rtk/generated/models';
import useModelOptions, {
  ComponentConstructorArgsConfigOfType,
  ScikitType,
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { useEffect, useMemo } from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import { StepFormFieldError, getStepSelectError } from './utils';
import { SimpleColumnConfig } from './types';

export default function SklearnModelInput() {
  const { getScikitOptions } = useModelOptions();
  const { control, trigger, watch, setValue } = useFormContext<ModelCreate>();

  const options = useMemo(() => {
    return getScikitOptions().map(toConstructorArgsConfig);
  }, [getScikitOptions]) as ComponentConstructorArgsConfigOfType<ScikitType>[];

  const config = watch('config') as SklearnModelSpec;

  const modelFitArgs: SklearnModelSchema['model']['fitArgs'] = useMemo(() => {
    if (config.framework !== 'sklearn') return { X: '', y: '' };

    const hasAnyFeaturizerOrTransform = (column: SimpleColumnConfig) => {
      return column.featurizers.length > 0 || column.transforms.length > 0;
    };

    const firstFeatureColumn = config.dataset.featureColumns[0];
    const firstTargetColumn = config.dataset.targetColumns[0];

    return {
      X: `$${firstFeatureColumn.name}${
        hasAnyFeaturizerOrTransform(firstFeatureColumn) ? '-out' : ''
      }`,
      y: `$${firstTargetColumn.name}${
        hasAnyFeaturizerOrTransform(firstTargetColumn) ? '-out' : ''
      }`,
    };
  }, [config]);

  useEffect(() => {
    if (config.spec.model) setValue('config.spec.model.fitArgs', modelFitArgs);
  }, [modelFitArgs]);

  return (
    <Controller
      control={control}
      name="config.spec"
      render={({ field, fieldState: { error } }) => (
        <PreprocessingStepSelect
          label="Sklearn Model"
          options={options}
          getError={getStepSelectError(
            //@ts-ignore
            () => error?.model as StepFormFieldError | undefined
          )}
          value={
            field.value ? (field.value as SklearnModelSchema).model : undefined
          }
          onChanges={(value) => {
            field.onChange({
              model: {
                ...value,
                fitArgs: modelFitArgs,
              } as SklearnModelSchema['model'],
            });
            trigger(field.name);
          }}
          onBlur={field.onBlur}
        />
      )}
    />
  );
}
