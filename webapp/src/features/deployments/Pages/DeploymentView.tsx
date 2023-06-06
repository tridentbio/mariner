import Content from '@components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import * as deploymentsApi from '@app/rtk/generated/deployments';
import Loading from '@components/molecules/Loading';
import { useEffect } from 'react';
import { DeploymentInferenceScreen } from '../Components/DeploymentInfereceScreen';
import { DeploymentHeader } from '../Components/DeploymentHeader';

const DeploymentView = () => {
  const deploymentIdMatch = useMatch('/deployments/:deploymentId');
  const deploymentId =
    deploymentIdMatch?.params.deploymentId &&
    parseInt(deploymentIdMatch.params.deploymentId);
  const [fetchDeployment, { data: deployment, isFetching }] =
    deploymentsApi.useLazyGetDeploymentQuery();

  useEffect(() => {
    fetchDeployment && deploymentId && fetchDeployment({ deploymentId });
  }, [deploymentId, fetchDeployment]);

  if (!deployment) {
    return <Loading isLoading={true} />;
  }

  return (
    <Content>
      <DeploymentHeader deployment={deployment} />
      <DeploymentInferenceScreen deployment={deployment} />
    </Content>
  );
};

export default DeploymentView;
