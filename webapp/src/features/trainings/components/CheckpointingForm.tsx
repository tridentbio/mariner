import { TrainingRequest } from '@app/rtk/generated/experiments';
import { TargetConfig } from '@app/rtk/generated/models';
import { MetricMode } from '@app/types/domain/experiments';
import { ModelVersionType } from '@app/types/domain/models';
import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import { required } from '@utils/reactFormRules';
import MetricSelect from './MetricSelect';
import ModeRadioInput from './ModeRadioInput';

const CheckpointingForm: React.FC<{ targetColumns: TargetConfig[] }> = ({
  targetColumns,
}) => {
  const { control, setValue } = useFormContext<TrainingRequest>();
  return (
    <>
      <Controller
        rules={{ ...required }}
        control={control}
        name="checkpointConfig.metricKey"
        render={({ field, fieldState: { error } }) => (
          <MetricSelect
            field={field}
            error={error}
            setValue={(value: MetricMode) =>
              setValue('checkpointConfig.mode', value)
            }
            targetColumns={targetColumns}
          />
        )}
      />
      <ModeRadioInput fieldName="checkpointConfig.mode" />
    </>
  );
};

export default CheckpointingForm;
