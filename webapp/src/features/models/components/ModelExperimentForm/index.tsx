import { ModelVersion } from '@app/rtk/generated/models';
import { BaseTrainingRequest } from '@app/types/domain/experiments';
import { APITargetConfig } from '@model-compiler/src/interfaces/torch-model-editor';
import {
  Button,
  Checkbox,
  FormControlLabel,
  TextField,
  Typography,
} from '@mui/material';
import { Box, SystemStyleObject } from '@mui/system';
import { deepClone } from '@utils';
import { useNotifications } from 'app/notifications';
import { Model } from 'app/types/domain/models';
import AdvancedOptionsForm from 'features/trainings/components/AdvancedOptionsForm';
import CheckpointingForm from 'features/trainings/components/CheckpointingForm';
import defaultExperimentFormValues from 'features/trainings/pages/CreateTraining/defaultExperimentFormValues';
import { useEffect, useState } from 'react';
import {
  Controller,
  ControllerRenderProps,
  DeepPartial,
  FormProvider,
  SubmitErrorHandler,
  SubmitHandler,
  useForm,
} from 'react-hook-form';
import { required } from 'utils/reactFormRules';
import ModelVersionSelect from '../ModelVersionSelect';
import OptimizerForm from './OptimizerForm';

export interface ModelExperimentFormProps {
  onSubmit: (value: BaseTrainingRequest) => any;
  onCancel?: () => any;
  initialValues?: DeepPartial<BaseTrainingRequest>;
  model: Model;
  loading?: Boolean;
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
    defaultValues: initialValues || {
      name: '',
    },
    shouldFocusError: true,
    criteriaMode: 'all',
    mode: 'onChange',
    shouldUnregister: true,
  });

  const { control, handleSubmit, watch, reset, register, unregister } = methods;

  //? make it be registered by the form as a field (when not declared by <Controller />).
  register('framework');

  const { notifyError } = useNotifications();

  const fieldStyle: SystemStyleObject = {
    mb: 1,
    width: '100%',
  };

  const onSubmitHandler: SubmitHandler<BaseTrainingRequest> = (training) => {
    if (!training.framework) throw new Error('Missing framework');

    onSubmit(training);
  };

  const onFormError: SubmitErrorHandler<BaseTrainingRequest> = (errors) => {
    notifyError('Resolve errors in the form');
  };

  const modelVersionFramework = watch('framework');

  useEffect(() => {
    if (initialValues && initialValues.modelVersionId) {
      const modelVersion = model.versions.find(
        (v) => v.id === initialValues.modelVersionId
      );

      if (modelVersion) {
        reset(modelVersion);

        setTargetColumns(
          modelVersion.config.dataset.targetColumns as APITargetConfig[]
        );
      }
    }
  }, [initialValues]);

  const handleModelVersionChange = (
    modelVersion: ModelVersion | undefined,
    field: ControllerRenderProps<BaseTrainingRequest, 'modelVersionId'>
  ) => {
    if (modelVersion) {
      setTargetColumns(
        modelVersion.config.dataset.targetColumns as APITargetConfig[]
      );

      reset((previousValue) => {
        if (modelVersion.config.framework == 'torch') {
          register('config');

          const previousTorchVersion =
            previousValue.framework == 'torch' && previousValue.modelVersionId;

          return {
            ...previousValue,
            modelVersionId: modelVersion.id,
            config: previousTorchVersion
              ? previousValue.config
              : deepClone(defaultExperimentFormValues.config || {}),
            framework: modelVersion.config.framework,
          } as BaseTrainingRequest;
        }

        unregister('config');

        return {
          name: previousValue.name,
          modelVersionId: modelVersion.id,
          framework: modelVersion.config.framework,
        };
      });
    } else {
      reset(
        (previousValue) =>
          ({
            name: previousValue.name,
          } as BaseTrainingRequest)
      );
    }
  };

  return (
    <FormProvider {...methods}>
      <form onSubmit={handleSubmit(onSubmitHandler, onFormError)}>
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
                  onChange={(modelVersion) =>
                    handleModelVersionChange(modelVersion, field)
                  }
                />
              )}
            />
          </Box>
          {modelVersionFramework === 'torch' && (
            <>
              <Controller
                rules={{ ...required }}
                control={control}
                name="config.optimizer"
                render={({ field, fieldState: { error } }) => (
                  <OptimizerForm
                    data-testid="optimizer"
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
                sx={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr 1fr',
                  gap: 2,
                }}
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
                <FormControlLabel
                  control={
                    <Controller
                      control={control}
                      defaultValue={false}
                      name="config.useGpu"
                      render={({ field }) => (
                        <Checkbox
                          {...field}
                          checked={field.value}
                          onChange={(e) => field.onChange(e.target.checked)}
                        />
                      )}
                    />
                  }
                  label="Use GPU?"
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
            </>
          )}
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
            <Button disabled={!!loading} variant="contained" type="submit">
              CREATE
            </Button>
          </Box>
        </Box>
      </form>
    </FormProvider>
  );
};

export default ModelExperimentForm;
