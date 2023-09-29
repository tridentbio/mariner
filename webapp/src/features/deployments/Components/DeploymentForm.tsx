import {
  Alert,
  InputLabel,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import { Box } from '@mui/system';
import MDEditor from '@uiw/react-md-editor';
import { Model } from 'app/types/domain/models';
import ModelVersionSelect from 'features/models/components/ModelVersionSelect';
import React, { useState } from 'react';
import { Controller, FormProvider, useForm } from 'react-hook-form';
import rehypeSanitize from 'rehype-sanitize';
import { required } from 'utils/reactFormRules';
import {
  DeploymentFormFields,
  ERateLimitUnits,
  EShareStrategies,
} from '../types';
import { Deployment } from 'app/rtk/generated/deployments';
import ShareStrategyInput from './ShareStrategyInput';
import SubmitCancelButtons from './SubmitCancelButtons';
import { yupResolver } from '@hookform/resolvers/yup';
import { deploymentFormSchema } from '../formSchema';
import PredictionsRateLimitInput from './PredictionsRateLimitInput';
import { useAppSelector } from 'app/hooks';
import ConfirmationDialog from 'components/templates/ConfirmationDialog';
import * as deploymentApi from 'app/rtk/generated/deployments';

type DeploymentFormProps = {
  model: Model;
  deployment?: Deployment;
  toggleModal: () => void;
};
const defaultFormState = {
  name: '',
  readme: '',
  shareStrategy: EShareStrategies.PRIVATE,
  usersIdAllowed: [],
  organizationsAllowed: [],
  predictionRateLimitValue: 10,
  predictionRateLimitUnit: ERateLimitUnits.MINUTE,
  showTrainingData: true,
};

const DeploymentForm: React.FC<DeploymentFormProps> = ({
  model,
  toggleModal,
}) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const currentDeployment = useAppSelector(
    (state) => state.deployments.current
  );

  const [createDeploy] = deploymentApi.useCreateDeploymentMutation();
  const [updateDeploy] = deploymentApi.useUpdateDeploymentMutation();
  const hookFormMethods = useForm<DeploymentFormFields>({
    defaultValues: currentDeployment || defaultFormState,
    reValidateMode: 'onChange',
    shouldUnregister: false,
    shouldFocusError: true,
    criteriaMode: 'all',
    mode: 'onChange',
    resolver: yupResolver(deploymentFormSchema),
  });
  const {
    control,
    register,
    handleSubmit,
    formState: { errors },
  } = hookFormMethods;

  const fieldStyle = {
    mb: 2,
    width: '100%',
  };

  const mutate = (value: DeploymentFormFields) => {
    if (currentDeployment) {
      updateDeploy({
        deploymentId: currentDeployment.id,
        deploymentUpdateInput: value,
      });
      return;
    }
    createDeploy({ deploymentBase: value });
  };
  const onSubmit = (value: DeploymentFormFields) => {
    if (value.shareStrategy === EShareStrategies.PUBLIC) {
      setShowConfirmation(true);
      return;
    }
    mutate(value);
    toggleModal();
  };

  const confirmPublicDeployment = () => {
    handleSubmit(mutate)();
    toggleModal();
  };

  return (
    <FormProvider {...hookFormMethods}>
      <ConfirmationDialog
        title="Confirm public deployment"
        text={'Are you sure to make this deployment public? '}
        alertText="Be aware that you will be responsible for usage charges incurred."
        onResult={(result) => {
          if (result === 'confirmed') confirmPublicDeployment();
          setShowConfirmation(false);
        }}
        open={showConfirmation}
      />
      <form onSubmit={handleSubmit(onSubmit)}>
        <Box data-color-mode="light">
          <TextField
            aria-label="deployment name"
            label="Deployment Name"
            error={!!errors.name}
            helperText={errors?.name?.message}
            sx={{ ...fieldStyle, mt: 1 }}
            {...register('name', { ...required })}
          />
          <Box sx={fieldStyle}>
            <InputLabel htmlFor="deployment-readme" sx={{ mb: 0.5 }}>
              Deployment README
            </InputLabel>
            <Controller
              control={control}
              name="readme"
              render={({ field, fieldState: { error } }) => (
                <MDEditor
                  previewOptions={{ rehypePlugins: [[rehypeSanitize]] }}
                  id="deployment-readme"
                  {...field}
                />
              )}
            />
          </Box>
          <Box sx={{ ...fieldStyle, mt: 3 }}>
            <Controller
              rules={{ ...required }}
              control={control}
              name="modelVersionId"
              render={({ field, fieldState: { error } }) => (
                <ModelVersionSelect
                  error={!!error}
                  helperText={error?.message}
                  model={model}
                  value={model.versions.find(
                    (version) => version.id === field.value
                  )}
                  onChange={(modelVersion) => field.onChange(modelVersion?.id)}
                />
              )}
            />
          </Box>
          <Box sx={fieldStyle}>
            <InputLabel htmlFor="shareStrategy">Share Strategy</InputLabel>
            <Alert
              variant="outlined"
              severity="warning"
              sx={{ mt: 0.5, fontSize: 14 }}
            >
              You are responsible for usage charges incurred by shared users.
            </Alert>
            <Controller
              control={control}
              name="shareStrategy"
              render={({ field, fieldState: { error } }) => (
                <ShareStrategyInput {...{ field }} />
              )}
            />
          </Box>
          <PredictionsRateLimitInput fieldStyle={fieldStyle} />
          <Box sx={fieldStyle}>
            <InputLabel>Display Training Data Distributions </InputLabel>
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography>No</Typography>
              <Controller
                control={control}
                name="showTrainingData"
                render={({ field }) => (
                  <Switch {...field} checked={field.value} />
                )}
              />
              <Typography>Yes</Typography>
            </Stack>
          </Box>
        </Box>
        <SubmitCancelButtons
          isNewDeployment={!currentDeployment}
          onCancel={toggleModal}
        />
      </form>
    </FormProvider>
  );
};

export { DeploymentForm };
