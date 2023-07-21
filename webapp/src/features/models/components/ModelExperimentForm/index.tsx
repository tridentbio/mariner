import { Button, TextField, Typography } from '@mui/material';
import { Box, SystemStyleObject } from '@mui/system';
import { useNotifications } from 'app/notifications';
import { FormEvent, useState } from 'react';
import { Model } from 'app/types/domain/models';
import ModelVersionSelect from '../ModelVersionSelect';
import { Controller, FormProvider, useForm } from 'react-hook-form';
import { required } from 'utils/reactFormRules';
import CheckpointingForm from 'features/trainings/components/CheckpointingForm';
import AdvancedOptionsForm from 'features/trainings/components/AdvancedOptionsForm';
import { DeepPartial } from 'react-hook-form';
import OptimizerForm from './OptimizerForm';
import { APITargetConfig } from '@model-compiler/src/interfaces/model-editor';
import { BaseTrainingRequest } from '@app/rtk/generated/experiments';

export interface ModelExperimentFormProps {
  onSubmit: (value: BaseTrainingRequest) => any;
  onCancel?: () => any;
  initialValues?: DeepPartial<BaseTrainingRequest>;
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
  const [targetColumns, setTargetColumns] = useState<APITargetConfig[]>(
    [] as APITargetConfig[]
  );

  const methods = useForm<BaseTrainingRequest>({
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
          const {
            config: { earlyStoppingConfig, ...restConfig },
          } = training;
          const payload = {
            ...training,
            config: restConfig,
          };
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
                        modelVersion.config.dataset
                          .targetColumns as APITargetConfig[]
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
            name="config.optimizer"
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
              name="config.batchSize"
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
              name="config.epochs"
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
          <CheckpointingForm
            targetColumns={targetColumns as APITargetConfig[]}
          />
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
