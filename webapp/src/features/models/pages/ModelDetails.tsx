import { useMatch } from 'react-router-dom';
import ModelDetailsView from '../components/ModelDetailsView';

const ModelDetails = () => {
  const detailsMatch = useMatch('/models/:modelName');
  const modelName =
    detailsMatch?.params.modelName && parseInt(detailsMatch?.params.modelName);

  if (!modelName) return null;

  return <ModelDetailsView modelId={modelName} />;
};

export default ModelDetails;
