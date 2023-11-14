import { APITargetConfig } from '@model-compiler/src/interfaces/torch-model-editor';
import { BaseTrainingRequest } from 'app/types/domain/experiments';
import React from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import { required } from 'utils/reactFormRules';
import MetricSelect from './MetricSelect';

const CheckpointingForm: React.FC<{ targetColumns: APITargetConfig[] }> = ({
  targetColumns,
}) => {
  const { control } = useFormContext<BaseTrainingRequest>();
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
            targetColumns={targetColumns}
          />
        )}
      />
    </>
  );
};

export default CheckpointingForm;
