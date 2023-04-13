import { Button, TextField, Typography } from '@mui/material';
import { Box, SystemStyleObject } from '@mui/system';
import { useNotifications } from '@app/notifications';
import { FormEvent, useState } from 'react';
import { Model, ModelVersionType } from '@app/types/domain/models';
import ModelVersionSelect from '../ModelVersionSelect';
import { Controller, FormProvider, useForm } from 'react-hook-form';
import { required } from 'utils/reactFormRules';
import CheckpointingForm from '@features/trainings/components/CheckpointingForm';
import AdvancedOptionsForm from '@features/trainings/components/AdvancedOptionsForm';
import { DeepPartial } from 'react-hook-form';
import { TrainingRequest } from '@app/rtk/generated/experiments';
import OptimizerForm from './OptimizerForm';
import { TargetConfig } from '@app/rtk/generated/models';

export interface ModelExperimentFormProps {
  onSubmit: (value: TrainingRequest) => any;
  onCancel?: () => any;
  initialValues?: DeepPartial<TrainingRequest>;
  model: Model;
  loading?: boolean;
}

const ModelExperimentForm = ({
  model,
  initialValues,
  onSubmit,
  onCancel,
  loading,
}: ModelExperimentFormProps) => {
  const [isUsingEarlyStopping, setIsUsingEarlyStopping] = useState(false);
  const [targetColumns, setTargetColumns] = useState<TargetConfig[]>(
    [] as TargetConfig[]
  );

  const methods = useForm<TrainingRequest>({
    defaultValues: initialValues,
    shouldFocusError: true,
    criteriaMode: 'all',
    mode: 'onChange',
    shouldUnregister: true,
  });
  const { control } = methods;

  const { notifyError } = useNotifications();

  const fieldStyle: SystemStyleObject = {
    mb: 1,
    width: '100%',
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();

    methods.handleSubmit(
      (training) => {
        if (!isUsingEarlyStopping) {
          const { earlyStoppingConfig, ...payload } = training;
          onSubmit(payload);
        } else {
          onSubmit(training);
        }
      },
      () => {
        notifyError('Resolve errors in the form');
      }
    )(event);
  };

  return (
    <FormProvider {...methods}>
      <form onSubmit={handleSubmit}>
        <Box sx={{ p: 0 }}>
          <Controller
            rules={{ ...required }}
            control={control}
            name="name"
            render={({ field, fieldState: { error } }) => (
              <TextField
                aria-label="experiment name"
                label="Experiment Name"
                error={!!error}
                helperText={error?.message}
                sx={fieldStyle}
                {...field}
              />
            )}
          />
          <Box sx={fieldStyle}>
            <Controller
              rules={{ ...required }}
              control={control}
              name="modelVersionId"
              render={({ field, fieldState: { error } }) => (
                <ModelVersionSelect
                  error={!!error}
                  helperText={error?.message}
                  model={model as Model}
                  value={model?.versions.find((v) => v.id === field.value)}
                  onChange={(modelVersion) => {
                    if (modelVersion)
                      setTargetColumns(
                        modelVersion.config.dataset.targetColumns
                      );
                    field.onChange({ target: { value: modelVersion?.id } });
                  }}
                />
              )}
            />
          </Box>

          <Controller
            rules={{ ...required }}
            control={control}
            name="optimizer"
            render={({ field, fieldState: { error } }) => (
              <OptimizerForm
                onChange={field.onChange}
                value={field.value}
                aria-label="optimizer"
                helperText={error?.message}
                label={error?.message || 'Optimizer'}
                error={!!error}
                sx={fieldStyle}
              />
            )}
          />
          <Box
            sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 2 }}
          >
            <Controller
              rules={{ ...required }}
              control={control}
              name="batchSize"
              render={({ field, fieldState: { error } }) => (
                <TextField
                  sx={fieldStyle}
                  aria-label="batch size"
                  label="Batch Size"
                  type="number"
                  inputProps={{ step: '1' }}
                  {...field}
                  helperText={error?.message}
                  error={!!error}
                />
              )}
            />

            <Controller
              rules={{ ...required }}
              control={control}
              name="epochs"
              render={({ field, fieldState: { error } }) => (
                <TextField
                  sx={fieldStyle}
                  aria-label="max epochs"
                  label="Epochs"
                  type="number"
                  {...field}
                  error={!!error}
                  helperText={error?.message}
                />
              )}
            />
          </Box>
          <Typography sx={{ mb: 1 }}>Checkpointing Options</Typography>
          <CheckpointingForm targetColumns={targetColumns} />
          <AdvancedOptionsForm
            open={isUsingEarlyStopping}
            onToggle={(value) => setIsUsingEarlyStopping(value)}
            targetColumns={targetColumns}
          />
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              '& > button': { margin: '0px 8px' },
            }}
          >
            {onCancel && (
              <Button variant="contained" color="error" onClick={onCancel}>
                PREVIOUS
              </Button>
            )}
            <Button disabled={loading} variant="contained" type="submit">
              CREATE
            </Button>
          </Box>
        </Box>
      </form>
    </FormProvider>
  );
};

export default ModelExperimentForm;
