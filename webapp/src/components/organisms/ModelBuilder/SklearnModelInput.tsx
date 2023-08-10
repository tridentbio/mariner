import { ModelCreate } from '@app/rtk/generated/models';
import useModelOptions, {
  ComponentConstructorArgsConfigOfType,
  ScikitType,
  toConstructorArgsConfig,
} from '@hooks/useModelOptions';
import { useMemo } from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import PreprocessingStepSelect from './PreprocessingStepSelect';

export default function SklearnModelInput() {
  const { getScikitOptions } = useModelOptions();
  const { control } = useFormContext<ModelCreate>();

  const options = useMemo(() => {
    return getScikitOptions().map(toConstructorArgsConfig);
  }, [getScikitOptions]) as ComponentConstructorArgsConfigOfType<ScikitType>[];

  return (
    <Controller
      control={control}
      name="config.spec"
      render={({ field }) => (
        <PreprocessingStepSelect
          label="Sklearn Model"
          options={options}
          value={(field.value as ScikitType | null) || undefined}
          onChanges={(value) => field.onChange(value)}
        />
      )}
    />
  );
}
