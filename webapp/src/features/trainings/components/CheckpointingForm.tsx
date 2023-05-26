import { BaseTrainingRequest } from 'app/rtk/generated/experiments';
import { TargetConfig } from 'app/rtk/generated/models';
import { MetricMode } from 'app/types/domain/experiments';
import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import { required } from 'utils/reactFormRules';
import MetricSelect from './MetricSelect';
import ModeRadioInput from './ModeRadioInput';

const CheckpointingForm: React.FC<{ targetColumns: TargetConfig[] }> = ({
  targetColumns,
}) => {
  const { control, setValue } = useFormContext<BaseTrainingRequest>();
  return (
    <>
      <Controller
        rules={{ ...required }}
        control={control}
        name="config.checkpointConfig.metricKey"
        render={({ field, fieldState: { error } }) => (
          <MetricSelect
            field={field}
            error={error}
            setValue={(value: MetricMode) =>
              setValue('config.checkpointConfig.mode', value)
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
