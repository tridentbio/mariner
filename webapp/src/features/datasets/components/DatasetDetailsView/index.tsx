import { ColumnMeta, DatasetErrors } from 'app/types/domain/datasets';
import { Button, CircularProgress, Container, Link } from '@mui/material';
import { LargerBoldText, Text } from '../../../../components/molecules/Text';
import { Box } from '@mui/system';
import DatasetStats from './DatasetStats';
import { useNavigate } from 'react-router-dom';
import { useNotifications } from '../../../../app/notifications';
import { humanFileSize } from '../../../../utils';

import * as datasetsApi from 'app/rtk/generated/datasets';
import { downloadDataset } from 'features/datasets/datasetSlice';
import { DatasetErrorsView } from './DatasetErrorsView';
import { lazy } from 'react';
import DataSummary from 'features/models/components/ModelVersionInferenceView/DataSummary';

const TorchModelEditorMarkdown = lazy(
  () => import('utils/codeSplittingAux/TorchModelEditorMarkdown')
);
interface DatasetDetailsProps {
  id: string;
}

const DatasetDetailsView = (props: DatasetDetailsProps) => {
  const { data: dataset } = datasetsApi.useGetMyDatasetQuery({
    datasetId: parseInt(props.id),
  });
  const [deleteDatasetById] = datasetsApi.useDeleteDatasetMutation();
  const navigate = useNavigate();
  const { setMessage } = useNotifications();

  if (!dataset) return <CircularProgress />;
  const handleDeleteDataset = async () => {
    const result = await deleteDatasetById({ datasetId: dataset.id });
    if ('data' in result && result.data) {
      setMessage({
        type: 'success',
        message: 'Deleted dataset',
      });
      navigate('/datasets');
    } else {
      setMessage({
        type: 'error',
        message: 'Failed to delete dataset',
      });
    }
  };

  const handleEdit = () => {
    navigate(`/datasets/${dataset.id}/edit`);
  };

  const toDict = (columnDescripns: ColumnMeta[]) => {
    const result: { [key: string]: { [key: string]: string } } = {};

    for (const col of columnDescripns) {
      if (!col.dataType.domainKind) continue;
      result[col.pattern] = {
        description: col.description,
        'data type': col.dataType.domainKind.toUpperCase(),
        unit: 'unit' in col.dataType ? col.dataType.unit : '',
      };
    }

    return result;
  };

  const handleCreateModelWithDataset = () => {
    navigate(`/models/new?datasetId=${dataset.id}`);
  };
  return (
    <Container>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingRight: 5,
        }}
      >
        <LargerBoldText
          id="dataset-name"
          maxWidth={'70%'}
          textOverflow="ellipsis"
          whiteSpace="nowrap"
          overflow="hidden"
        >
          {dataset.name}
        </LargerBoldText>
        <div>
          <Button
            onClick={handleCreateModelWithDataset}
            variant="contained"
            color="primary"
          >
            New Model
          </Button>
          <Button
            variant="contained"
            color="primary"
            sx={{ ml: 3 }}
            onClick={handleEdit}
          >
            Edit
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleDeleteDataset}
            id="delete-dataset"
            sx={{ ml: 3 }}
          >
            Delete
          </Button>
        </div>
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'row', mt: 5 }}>
        <Box sx={{ mr: 2 }}>
          <Text>CSV:</Text>
          <Text>Rows:</Text>
          <Text>Columns:</Text>
          <Text>Size:</Text>
          <Text>Split:</Text>
        </Box>
        <Box>
          <Link
            sx={{ cursor: 'pointer' }}
            onClick={() =>
              dataset.dataUrl && downloadDataset(dataset.id, dataset.dataUrl)
            }
          >
            <Text>{dataset.dataUrl}</Text>
          </Link>
          <Text>{dataset.rows || ''}</Text>
          <Text>{dataset.columns || ''}</Text>
          <Text>{dataset.bytes && humanFileSize(dataset.bytes)}</Text>
          <Text id="dataset-split">
            {dataset.splitTarget} ({dataset.splitType})
          </Text>
        </Box>
      </Box>

      <LargerBoldText mt={3} mb={3}>
        Dataset Overview
      </LargerBoldText>
      <div id="dataset-description">
        <TorchModelEditorMarkdown
          source={dataset.description}
          warpperElement={{
            'data-color-mode': 'light',
          }}
        />
      </div>

      <LargerBoldText>Column Descriptions</LargerBoldText>
      {dataset.columnsMetadata && (
        <DatasetStats stats={toDict(dataset.columnsMetadata)} />
      )}

      {dataset.readyStatus === 'failed' && dataset.errors && (
        <DatasetErrorsView
          errors={dataset.errors as unknown as DatasetErrors}
          datasetId={dataset.id}
        />
      )}

      {dataset.readyStatus === 'ready' && (
        <>
          <LargerBoldText mt={3}>Dataset Statistics</LargerBoldText>
          <DataSummary columnsData={dataset.stats}></DataSummary>
        </>
      )}

      {dataset.readyStatus === 'processing' && (
        <>
          <LargerBoldText mt={3}>Processing</LargerBoldText>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'center',
              mt: 5,
            }}
          >
            <CircularProgress sx={{ mt: 3 }} size={100} />
          </Box>
        </>
      )}
    </Container>
  );
};

export default DatasetDetailsView;
