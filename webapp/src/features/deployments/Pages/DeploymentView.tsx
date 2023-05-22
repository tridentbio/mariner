import Content from '@components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import * as deploymentsApi from '@app/rtk/generated/deployments';
import { Box, Divider, SxProps, Typography } from '@mui/material';
import { LargerBoldText } from '@components/molecules/Text';
import Loading from '@components/molecules/Loading';
import ModalHeader from '@components/templates/Modal/ModalHeader';
import ModelEditorMarkdown from '@utils/codeSplittingAux/ModelEditorMarkdown';
import { Section } from '@components/molecules/Section';
import { DeploymentPrediction } from '@components/templates/DeploymentPrediction';
import StatusChip from '../Components/StatutsChip';
import DataSummary, {
  DataSummaryProps,
} from '@features/models/components/ModelVersionInferenceView/DataSummary';
import { DeploymentWithTrainingData } from '@app/rtk/generated/deployments';
import { useEffect } from 'react';

const DeploymentHeader = ({
  deployment,
}: {
  deployment: DeploymentWithTrainingData;
}) => (
  <>
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <ModalHeader>
        <LargerBoldText mr="auto">{deployment.name}</LargerBoldText>
      </ModalHeader>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
          alignItems: 'flex-end',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <StatusChip
            sx={{
              textTransform: 'uppercase',
              fontWeight: 'bold',
              padding: '0.5rem',
            }}
            status={deployment.status}
          />
        </Box>
        <Box
          sx={{
            mb: '1rem',
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <Typography>
            Rate Limit:{' '}
            <b>{`${deployment.predictionRateLimitValue}/${deployment.predictionRateLimitUnit}`}</b>
          </Typography>
        </Box>
      </Box>
    </Box>

    <Divider sx={{ mb: '1rem' }} />
  </>
);

const readmeSx: SxProps = {
  mb: '1rem',
  mt: '1rem',
  ml: '5px',
  border: '1px solid rgba(0, 0, 0, 0.12)',
  padding: '1rem',
  borderRadius: '4px',
};

const DeploymentView = () => {
  const deploymentIdMatch = useMatch('/deployments/:deploymentId');
  const deploymentId =
    deploymentIdMatch?.params.deploymentId &&
    parseInt(deploymentIdMatch.params.deploymentId);
  const [fetchDataset, { data: deployment, isFetching }] =
    deploymentsApi.useLazyGetDeploymentQuery();

  useEffect(() => {
    fetchDataset && deploymentId && fetchDataset({ deploymentId });
  }, [deploymentId, fetchDataset]);

  if (!deployment) {
    return <Loading isLoading={true} />;
  }
  console.log({ deployment });
  return (
    <Content>
      <DeploymentHeader deployment={deployment} />
      <Section title="Readme">
        <Box sx={readmeSx}>
          <ModelEditorMarkdown
            source={deployment.readme}
            warpperElement={{
              'data-color-mode': 'light',
            }}
          />
        </Box>
      </Section>
      <DeploymentPrediction deployment={deployment} />
      {deployment.trainingData?.datasetSummary && (
        <Section title="Training Data">
          <DataSummary
            columnsData={
              deployment.trainingData
                .datasetSummary as DataSummaryProps['columnsData']
            }
          />
        </Section>
      )}
    </Content>
  );
};

export default DeploymentView;
