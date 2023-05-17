import Content from '@components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import { deploymentsApi } from '../deploymentsApi';
import { Box } from '@mui/material';
import { Text } from '@components/molecules/Text';
import Loading from '@components/molecules/Loading';

type SectionProps = {
  title: string;
  children: React.ReactNode;
};

const Section = ({ children, title, ...rest }: SectionProps) => {
  return (
    <Box sx={{ mb: 1 }}>
      <Text fontWeight="bold">{title}:</Text>
      <Box sx={{ ml: 1 }} {...rest}>
        {children}
      </Box>
    </Box>
  );
};

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
      <Box>
        <Section title="Name">
          <Text>{deployment.name}</Text>
        </Section>
        <Section title="Readme">
          <Text>{deployment.readme}</Text>
        </Section>
      </Box>
    </Content>
  );
};

export default DeploymentView;
