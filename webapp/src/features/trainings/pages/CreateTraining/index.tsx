import { Box, Step, StepContent, StepLabel, Stepper } from '@mui/material';
import { useNotifications } from 'app/notifications';
import * as experimentsApi from 'app/rtk/generated/experiments';
import { Model } from 'app/types/domain/models';
import { BaseTrainingRequest } from 'app/types/domain/experiments';
import Content from 'components/templates/AppLayout/Content';
import ModelsSelect from 'components/atoms/ModelsSelect';
import ProcessingModal from 'components/organisms/ProcessingModal';
import ModelExperimentForm from 'features/models/components/ModelExperimentForm';
import { addTraining } from 'features/models/modelSlice';
import { useState } from 'react';
import { useDispatch } from 'react-redux';
import { useNavigate } from 'react-router-dom';

const CreateTraining: React.FC = () => {
  const [startTraining, { isLoading }] =
    experimentsApi.usePostExperimentsMutation();
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const { notifyError, success } = useNotifications();
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const handleStartTraning = async (exp: BaseTrainingRequest) => {
    const { config, ...experimentPayload } = exp;
    const payload = (
      exp.framework === 'sklearn'
        ? {
            ...experimentPayload,
          }
        : exp
    ) as BaseTrainingRequest;
    await startTraining({
      body: payload,
    })
      .unwrap()
      .then((newExp) => {
        // @ts-ignore
        dispatch(addTraining(newExp));
        success('Experiment started');
        navigate(`/models/${selectedModel?.id}?#newtraining`);
      })
      .catch(() => notifyError('Experiment failed to start'));
  };

  const steps = [
    {
      label: `Model${selectedModel ? ': ' + selectedModel.name : ''}`,
      description: ``,
      content: (
        <Box>
          <ModelsSelect
            onChange={(value) => {
              setSelectedModel(value);
              if (value) {
                setActiveStep(1);
              }
            }}
          />
        </Box>
      ),
    },
    {
      label: 'Training Options',
      description: '',
      content: (
        <>
          {selectedModel && (
            <ModelExperimentForm
              model={selectedModel}
              onSubmit={handleStartTraning}
              onCancel={() => setActiveStep(0)}
            />
          )}
        </>
      ),
    },
  ];
  return (
    <>
      <ProcessingModal processing={isLoading} type="Processing" />
      <Content>
        <Stepper activeStep={activeStep} orientation="vertical">
          {steps.map((step, index) => (
            <Step key={step.label}>
              <StepLabel>{step.label}</StepLabel>
              <StepContent>{step.content}</StepContent>
            </Step>
          ))}
        </Stepper>
      </Content>
    </>
  );
};

export default CreateTraining;
