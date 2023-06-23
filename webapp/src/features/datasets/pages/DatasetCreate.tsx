import { useNotifications } from 'app/notifications';
import DatasetForm from '../components/DatasetForm';
import { useNavigate } from 'react-router-dom';
import Content from 'components/templates/AppLayout/Content';
import { datasetsApi } from 'app/rtk/datasets';
import { NewDataset } from 'app/types/domain/datasets';
import { isApiError, messageApiError } from 'app/rtk/api';
import { gzipCompress } from 'utils/gzipCompress';

const DatasetCreate = ({ onCreate }: { onCreate: CallableFunction }) => {
  const [createDataset, { isLoading }] = datasetsApi.usePostDatasetsMutation();
  const { setMessage } = useNotifications();
  const navigate = useNavigate();

  const handleCreateDataset = async (datasetCreate: NewDataset) => {
    try {
      // @ts-ignore
      datasetCreate.file = await gzipCompress(datasetCreate.file);
      await createDataset(datasetCreate).unwrap();
      setMessage({ type: 'success', message: 'Dataset created!' });
      onCreate();
    } catch (error) {
      if (isApiError(error))
        setMessage({ type: 'error', message: messageApiError(error) });
    }
  };

  return (
    <Content disableBreadcrumb>
      <DatasetForm
        loading={isLoading}
        onSubmit={handleCreateDataset}
        onCancel={() => navigate(-1)}
      />
    </Content>
  );
};

export default DatasetCreate;
