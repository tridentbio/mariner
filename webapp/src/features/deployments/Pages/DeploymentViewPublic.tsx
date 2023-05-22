import { useMatch } from 'react-router-dom';
import { DeploymentScreen } from './DeploymentView';
import * as deploymentsApi from '@app/rtk/generated/deployments';
import NotFound from '@components/atoms/NotFound';

const DeploymentViewPublic = () => {
  const deploymentTokenMatch = useMatch(
    '/public-model/:token1/:token2/:token3'
  );
  const params = deploymentTokenMatch?.params || {};
  const tokens = Object.values(params);

  const { data: deployment } = deploymentsApi.useGetPublicDeploymentQuery({
    token: tokens && tokens.length === 3 ? tokens.join('.') : '',
  });

  if (!deployment) return <NotFound>Deployment not found</NotFound>;

  return <DeploymentScreen deployment={deployment} publicDeployment />;
};

export default DeploymentViewPublic;
