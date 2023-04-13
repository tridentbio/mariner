import Content from 'components/templates/AppLayout/Content';
import { useMatch } from 'react-router-dom';
import ModelVersionDetailsView from '../components/ModelVersionDetailsView';

const ModelVersionDetails = () => {
  const versionDetailsMatch = useMatch('/models/:modelName/:modelVersion');
  const modelVersionId =
    versionDetailsMatch?.params.modelVersion &&
    parseInt(versionDetailsMatch.params.modelVersion);
  const modelId =
    versionDetailsMatch?.params.modelName &&
    parseInt(versionDetailsMatch.params.modelName);
  return modelVersionId && modelId ? (
    <Content>
      <ModelVersionDetailsView
        modelVersionId={modelVersionId}
        modelId={modelId}
      />
    </Content>
  ) : null;
};

export default ModelVersionDetails;
