import { Button } from '@mui/material';
import Modal from 'components/templates/Modal';
import Content from 'components/templates/AppLayout/Content';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DatasetTable from '../components/DatasetTable';
import DatasetCreate from './DatasetCreate';

const DatasetListing = () => {
  const navigate = useNavigate();
  const [creatingDataset, setCreatingDataset] = useState(false);
  return (
    <Content>
      <Button
        variant="contained"
        color="primary"
        style={{ margin: '5px 0 5px 0' }}
        onClick={() => setCreatingDataset(true)}
        id="go-to-create-dataset"
      >
        Add Dataset
      </Button>
      <DatasetTable
        onOpenDetails={(datasetId) => navigate(`/datasets/${datasetId}`)}
      />
      <Modal
        title={'Create Dataset'}
        open={creatingDataset}
        onClose={() => setCreatingDataset(false)}
      >
        <DatasetCreate onCreate={() => setCreatingDataset(false)} />
      </Modal>
    </Content>
  );
};

export default DatasetListing;
