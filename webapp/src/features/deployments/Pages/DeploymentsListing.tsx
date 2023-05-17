import { Box, Button } from '@mui/material';
import DeploymentsTable from '../Components/DeploymentsTable';
import Content from '@components/templates/AppLayout/Content';

const DeploymentsListing = () => {
  return (
    <Content>
      <Box sx={{ ml: 'auto', mr: 'auto' }}>
        <DeploymentsTable />
      </Box>
    </Content>
  );
};

export default DeploymentsListing;
