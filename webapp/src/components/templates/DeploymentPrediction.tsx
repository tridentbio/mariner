import { Deployment } from '@app/rtk/generated/deployments';
import { InferenceInputs } from '@components/organisms/InferenceInput';
import { Box, Button } from '@mui/material';
import { useMemo, useState } from 'react';
import { ModelOutputValue } from '@app/types/domain/models';
import api from '@app/api';
import { useNotifications } from '@app/notifications';
import { Text } from '@components/molecules/Text';
import { InferenceOutput } from '@components/organisms/InferenceOutput';
import { APITargetConfig } from '@model-compiler/src/interfaces/torch-model-editor';
import {
  ColumnConfig,
  ColumnConfigWithPreprocessing,
} from '@app/rtk/generated/models';

const getPredictionPrivate = async (
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

const getPredictionPublic = async (
  deployment: Deployment,
  inputValues: Record<string, string | number | string[] | number[]>
) => {
  Object.keys(inputValues).forEach((key) => {
    if (!Array.isArray(inputValues[key]))
      // @ts-ignore
      inputValues[key] = [inputValues[key]];
  });
  const response = await api.post(
    `api/v1/deployments/${deployment.id}/predict-public`,
    inputValues
  );
  return response.data;
};

export const DeploymentPrediction = ({
  deployment,
  publicDeployment,
}: {
  deployment: Deployment;
  publicDeployment: boolean;
}) => {
  const getPrediction = publicDeployment
    ? getPredictionPublic
    : getPredictionPrivate;

  const inferenceColumns:
    | (ColumnConfig | ColumnConfigWithPreprocessing)[]
    | undefined = useMemo(
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
    inferenceColumns?.reduce<{ [key: string]: string | number }>(
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

    if (
      Object.values(inputValues).some((v: any) =>
        [null, undefined, ''].includes(v)
      )
    )
      return setOutputValues(null);

    getPrediction(deployment, inputValues)
      .then(setOutputValues)
      .catch(
        (err) =>
          err.response?.data?.detail &&
          setMessage({ message: err.response.data.detail, type: 'error' })
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
        <Text fontWeight="bold">Input:</Text>
        <Button
          onClick={handlePrediction}
          variant="contained"
          color="primary"
          sx={{ ml: 3 }}
        >
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
          targetColumns={targetColumns as APITargetConfig[]}
        />
      )}
    </Box>
  );
};
