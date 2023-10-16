import ShareStrategyInput from '@features/deployments/Components/ShareStrategyInput';
import SubmitCancelButtons from '@features/deployments/Components/SubmitCancelButtons';
import { EShareStrategies } from '@features/deployments/types';
import {
  InputLabel,
  TextField
} from '@mui/material';
import { Box } from '@mui/system';
import MDEditor from '@uiw/react-md-editor';
import { useAppSelector } from 'app/hooks';
import { Deployment } from 'app/rtk/generated/deployments';
import { Model } from 'app/types/domain/models';
import ConfirmationDialog from 'components/templates/ConfirmationDialog';
import React, { useState } from 'react';
import { Controller, FormProvider, useForm } from 'react-hook-form';
import rehypeSanitize from 'rehype-sanitize';
import { required } from 'utils/reactFormRules';

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
  showTrainingData: true,
};

export const ModelTemplateForm: React.FC<DeploymentFormProps> = ({
  model,
  toggleModal,
}) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const currentDeployment = useAppSelector(
    (state) => state.deployments.current
  );

  const hookFormMethods = useForm<DeploymentFormFields>({
    defaultValues: currentDeployment || defaultFormState,
    reValidateMode: 'onChange',
    shouldUnregister: false,
    shouldFocusError: true,
    criteriaMode: 'all',
    mode: 'onChange',
    // resolver: yupResolver(deploymentFormSchema),
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

  const onSubmit = (value: DeploymentFormFields) => {
    if (value.shareStrategy === EShareStrategies.PUBLIC) {
      setShowConfirmation(true);
      return;
    }

    // createDeploy({ deploymentBase: value });

    toggleModal();
  };

  const confirmPublicDeployment = () => {
    // handleSubmit(mutate)();
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
            aria-label="Template name"
            label="Template name"
            error={!!errors.name}
            // helperText={errors?.name?.message}
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
          <Box sx={fieldStyle}>
            <InputLabel htmlFor="shareStrategy">Share Strategy</InputLabel>
            <Controller
              control={control}
              name="shareStrategy"
              render={({ field, fieldState: { error } }) => (
                <ShareStrategyInput {...{ field }} />
              )}
            />
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
