import { Box, Button } from '@mui/material';
import NotFound from 'components/atoms/NotFound';
import AppTabs, { AppTabsProps } from 'components/organisms/Tabs';
import { LargerBoldText } from 'components/molecules/Text';
import { useNotifications } from 'app/notifications';
import ModelExperiments from './ModelExperiments';
import ModalHeader from 'components/templates/Modal/ModalHeader';
import { useNavigate, useLocation } from 'react-router-dom';
import Content from 'components/templates/AppLayout/Content';
import ModelInferenceView from './ModelInferenceView';
import { modelsApi } from 'app/rtk/models';
import ModelOverview from './ModelOverview';
import Loading from 'components/molecules/Loading';
import ModelMetricsView from './ModelMetricsView';
import { ModelDeployments } from '@features/deployments/Pages/ModelDeployments';

interface ModelDetailsProps {
  modelId: number;
}

const ModelDetailsView = ({ modelId }: ModelDetailsProps) => {
  const { data: model, isLoading: isModelLoading } =
    modelsApi.useGetModelByIdQuery(modelId);
  const [deleteModel, { isLoading: isDeleting }] =
    modelsApi.useDeleteModelOldMutation();
  const { hash } = useLocation();
  const { notifyError: error, success } = useNotifications();
  const navigate = useNavigate();

  const handleDeleteModel = async () => {
    if (!model) return;
    try {
      await deleteModel(model.id);
      success('Model deleted');
      navigate(-1);
    } catch {
      error('Failed to delete model');
    }
  };

  const handleAddVersion = async () => {
    model && navigate(`/models/new?registeredModel=${model.id}`);
  };

  const tabs: AppTabsProps['tabs'] = [
    {
      label: 'Model',
      panel: model && <ModelOverview model={model} />,
    },
    {
      label: 'Training',
      panel: model && (
        <>
          <Box>
            <Button
              variant="contained"
              color="primary"
              onClick={() => navigate(`/trainings/new?modelId=${model.id}`)}
            >
              Create Training
            </Button>
            <ModelExperiments model={model} />
          </Box>
        </>
      ),
    },
    {
      label: 'Inference',
      panel: model && (
        <Box>
          <ModelInferenceView model={model} />
        </Box>
      ),
    },
    {
      label: 'Metrics',
      panel: model?.versions.length && (
        <Box>
          <ModelMetricsView model={model} />
        </Box>
      ),
    },
    {
      label: 'Deployments',
      panel: model?.versions.length && (
        <Box>
          <ModelDeployments model={model} />
        </Box>
      ),
    },
  ];

  if (!isModelLoading && !model) return <NotFound>Model not found</NotFound>;

  return (
    <Content>
      <Loading isLoading={isModelLoading} />
      {model && (
        <>
          <ModalHeader>
            <Box sx={{ display: 'flex', flexDirection: 'row', marginRight: 5 }}>
              <LargerBoldText mr="auto">{model.name}</LargerBoldText>
              <Button
                onClick={handleAddVersion}
                variant="contained"
                color="primary"
                sx={{ ml: 3 }}
              >
                ADD VERSION
              </Button>
              <Button
                variant="contained"
                color="error"
                sx={{ ml: 3 }}
                onClick={handleDeleteModel}
                disabled={isDeleting}
              >
                Delete
              </Button>
            </Box>
          </ModalHeader>
          <AppTabs initialTab={hash === '#newtraining' ? 1 : 0} tabs={tabs} />
        </>
      )}
    </Content>
  );
};

export default ModelDetailsView;
