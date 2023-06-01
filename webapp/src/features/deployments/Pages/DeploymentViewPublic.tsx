import { useMatch } from 'react-router-dom';
import * as deploymentsApi from '@app/rtk/generated/deployments';
import NotFound from '@components/atoms/NotFound';
import Content from '@components/templates/AppLayout/Content';
import { DeploymentHeader } from '../Components/DeploymentHeader';
import { DeploymentInferenceScreen } from '../Components/DeploymentInfereceScreen';
import { useAppSelector } from '@app/hooks';

const DeploymentViewPublic = () => {
  const deploymentTokenMatch = useMatch(
    '/public-model/:token1/:token2/:token3'
  );
  const params = deploymentTokenMatch?.params || {};
  const tokens = Object.values(params);

  deploymentsApi.useGetPublicDeploymentQuery({
    token: tokens && tokens.length === 3 ? tokens.join('.') : '',
  });

  const deployment = useAppSelector((state) => state.deployments.current);

  if (!deployment) return <NotFound>Deployment not found</NotFound>;

  return (
    <Content>
      <DeploymentHeader deployment={deployment} />
      <DeploymentInferenceScreen deployment={deployment} publicDeployment />
    </Content>
  );
};

export default DeploymentViewPublic;
