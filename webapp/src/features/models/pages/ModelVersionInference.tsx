import { CircularProgress } from '@mui/material';
import Content from 'components/templates/AppLayout/Content';
import NotFound from 'components/atoms/NotFound';
import { useEffect } from 'react';
import { useMatch } from 'react-router-dom';
import ModelVersionInferenceView from '../components/ModelVersionInferenceView';
import { modelsApi } from 'app/rtk/models';

const VersionNotFound = () => <NotFound>Model version not found :( </NotFound>;

const ModelVersionInference = () => {
  const match = useMatch('/models/:modelName/:modelVersion/inference');
  const { modelName, modelVersion } = match?.params || {};
  const [fetchSingleModel, { isLoading: fetchingModel, data: model }] =
    modelsApi.useLazyGetModelByIdQuery();
  useEffect(() => {
    const modelId = modelName && parseInt(modelName);
    if (modelId) {
      fetchSingleModel(modelId);
    }
  }, [modelName]);

  const modelVersionId = modelVersion && parseInt(modelVersion);

  return (
    <Content>
      {(!modelName || !modelVersion) && <VersionNotFound />}
      {fetchingModel && <CircularProgress />}
      {model && modelVersionId && (
        <ModelVersionInferenceView
          model={model}
          modelVersionId={modelVersionId}
        />
      )}
    </Content>
  );
};

export default ModelVersionInference;
