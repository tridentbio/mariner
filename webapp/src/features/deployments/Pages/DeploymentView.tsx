import Content from '@components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import { deploymentsApi } from '../deploymentsApi';
import { Box, Divider } from '@mui/material';
import { LargerBoldText } from '@components/molecules/Text';
import Loading from '@components/molecules/Loading';
import ModalHeader from '@components/templates/Modal/ModalHeader';
import ModelEditorMarkdown from '@utils/codeSplittingAux/ModelEditorMarkdown';
import { Section } from '@components/molecules/Section';
import { DeploymentPrediction } from '@components/templates/DeploymentPrediction';

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
      <ModalHeader>
        <LargerBoldText mr="auto">{deployment.name}</LargerBoldText>
      </ModalHeader>
      <Divider sx={{ mb: '1rem' }} />
      <Section title="Readme">
        <Box
          sx={{
            mb: '1rem',
            mt: '1rem',
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
