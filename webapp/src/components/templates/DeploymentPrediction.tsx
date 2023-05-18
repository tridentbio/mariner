import { Deployment } from '@app/rtk/generated/deployments';
import { InferenceInputs } from '@components/organisms/InferenceInput';
import { Box, Button } from '@mui/material';
import { useMemo, useState } from 'react';
import { ModelOutputValue } from '@app/types/domain/models';
import api from '@app/api';
import { useNotifications } from '@app/notifications';
import { Text } from '@components/molecules/Text';
import { InferenceOutput } from '@components/organisms/InferenceOutput';

const getPrediction = async (
  deployment: Deployment,
  inputValues: Record<string, string | number | string[] | number[]>
) => {
  Object.keys(inputValues).forEach((key) => {
    if (!Array.isArray(inputValues[key]))
      // @ts-ignore
      inputValues[key] = [inputValues[key]];
  });

  const response = await api.post(
    `api/v1/deployments/${deployment.id}/predict`,
    inputValues
  );
  return response.data;
};

export const DeploymentPrediction = ({
  deployment,
}: {
  deployment: Deployment;
}) => {
  const inferenceColumns = useMemo(
    () => deployment.modelVersion?.config.dataset?.featureColumns,
    [deployment.id]
  );
  const targetColumns = useMemo(
    () => deployment.modelVersion?.config.dataset?.targetColumns,
    [deployment.id]
  );
  if (!inferenceColumns || !targetColumns) return null;

  const [inputValues, setInputValues] = useState<{
    [key: string]: string | number;
  }>(
    inferenceColumns?.reduce(
      (acc, column) => ({ ...acc, [column.name]: '' }),
      {}
    )
  );
  const handleInputValues = (key: string, value: string | number) =>
    setInputValues({ ...inputValues, [key]: value });

  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const { setMessage } = useNotifications();
  const [outputValues, setOutputValues] = useState<ModelOutputValue | null>(
    null
  );
  const handlePrediction = () => {
    if (loadingPrediction) return;
    setLoadingPrediction(true);

    if (Object.values(inputValues).some((v) => !v))
      return setOutputValues(null);

    getPrediction(deployment, inputValues)
      .then(setOutputValues)
      .catch(
        (err) =>
          err.data?.detail &&
          setMessage({ message: err.data.detail, type: 'error' })
      )
      .finally(() => setLoadingPrediction(false));
  };

  return (
    <Box>
      <Box
        sx={{
          mb: 1,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Text fontWeight="bold">Prediction:</Text>
        <Button onClick={handlePrediction} sx={{ ml: 1, p: '1rem' }}>
          Predict
        </Button>
      </Box>
      <InferenceInputs
        inferenceColumns={inferenceColumns}
        handleChange={handleInputValues}
        values={inputValues}
      />
      {outputValues && (
        <InferenceOutput
          outputValues={outputValues}
          targetColumns={targetColumns}
        />
      )}
    </Box>
  );
};
