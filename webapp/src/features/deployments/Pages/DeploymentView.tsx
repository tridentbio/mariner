import Content from '@components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import { deploymentsApi } from '../deploymentsApi';
import { Box, Divider, Typography } from '@mui/material';
import { LargerBoldText } from '@components/molecules/Text';
import Loading from '@components/molecules/Loading';
import ModalHeader from '@components/templates/Modal/ModalHeader';
import ModelEditorMarkdown from '@utils/codeSplittingAux/ModelEditorMarkdown';
import { Section } from '@components/molecules/Section';
import { DeploymentPrediction } from '@components/templates/DeploymentPrediction';
import StatusChip from '../Components/StatutsChip';

const DeploymentView = () => {
  const deploymentIdMatch = useMatch('/deployments/:deploymentId');
  const deploymentId =
    deploymentIdMatch?.params.deploymentId &&
    parseInt(deploymentIdMatch.params.deploymentId);
  const deployment =
    deploymentId && deploymentsApi.useGetDeploymentByIdQuery(deploymentId).data;
  if (!deployment) {
    return <Loading isLoading={true} />;
  }

  return (
    <Content>
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

      <Section title="Readme">
        <Box
          sx={{
            mb: '1rem',
            mt: '1rem',
            ml: '5px',
            border: '1px solid rgba(0, 0, 0, 0.12)',
            padding: '1rem',
            borderRadius: '4px',
          }}
        >
          <ModelEditorMarkdown
            source={deployment.readme}
            warpperElement={{
              'data-color-mode': 'light',
            }}
          />
        </Box>
      </Section>
      <DeploymentPrediction deployment={deployment} />
    </Content>
  );
};

export default DeploymentView;
