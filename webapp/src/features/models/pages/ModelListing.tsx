import { Button } from '@mui/material';
import Content from 'components/templates/AppLayout/Content';
import { useNavigate } from 'react-router-dom';
import ModelTable from '../components/ModelTable';

const ModelListing = () => {
  const navigate = useNavigate();

  return (
    <Content>
      <Button
        sx={{
          margin: '5px 0 5px 0',
        }}
        variant="contained"
        onClick={() => navigate('/models/new')}
      >
        Create model
      </Button>
      <ModelTable onOpenDetails={(model) => navigate(`/models/${model.id}`)} />
    </Content>
  );
};

export default ModelListing;
