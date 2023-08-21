import { ModelCreate, SklearnModelSchema } from '@app/rtk/generated/models';
import useModelOptions, {
  ComponentConstructorArgsConfigOfType,
  ScikitType,
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { useMemo } from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import PreprocessingStepSelect from './PreprocessingStepSelect';
import { StepFormFieldError, getStepSelectError } from './utils';

export default function SklearnModelInput() {
  const { getScikitOptions } = useModelOptions();
  const { control, trigger } = useFormContext<ModelCreate>();

  const options = useMemo(() => {
    return getScikitOptions().map(toConstructorArgsConfig);
  }, [getScikitOptions]) as ComponentConstructorArgsConfigOfType<ScikitType>[];

  return (
    <Controller
      control={control}
      name="config.spec"
      render={({ field, fieldState: { error } }) => (
        <PreprocessingStepSelect
          label="Sklearn Model"
          testId="sklearn-model-select"
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
