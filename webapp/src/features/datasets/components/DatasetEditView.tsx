import { CircularProgress } from '@mui/material';
import DatasetForm from './DatasetForm';
import { UpdateDataset } from 'app/types/domain/datasets';
import { useNavigate } from 'react-router-dom';
import { useNotifications } from '../../../app/notifications';
import * as datasetsApi from 'app/rtk/generated/datasets';

interface DatasetEditProps {
  id: string;
}
const DatasetEditView = (props: DatasetEditProps) => {
  const { data: dataset } = datasetsApi.useGetMyDatasetQuery({
    datasetId: parseInt(props.id),
  });
  const [updateDataset] = datasetsApi.useUpdateDatasetMutation();
  const navigate = useNavigate();
  const { setMessage } = useNotifications();
  if (!dataset) return <CircularProgress />;

  const handleUpdate = async (dataset: Omit<UpdateDataset, 'datasetId'>) => {
    return updateDataset({
      datasetUpdateInput: dataset,
      datasetId: Number(props.id),
    })
      .unwrap()
      .then((result) => {
        setMessage({ message: 'Dataset updated', type: 'success' });
        navigate(`/datasets/${result.id}`);
      })
      .catch(() => {
        setMessage({ message: 'Failed to update dataset', type: 'error' });
      });
  };

  return (
    <DatasetForm
      onSubmit={handleUpdate}
      initialValues={dataset}
      onCancel={() => navigate(-1)}
    />
  );
};

export default DatasetEditView;
