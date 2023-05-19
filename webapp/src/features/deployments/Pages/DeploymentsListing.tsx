import { Box, Divider } from '@mui/material';
import DeploymentsTable from '../Components/DeploymentsTable';
import Content from '@components/templates/AppLayout/Content';
import { LargerBoldText } from '@components/molecules/Text';
import ModalHeader from '@components/templates/Modal/ModalHeader';

const DeploymentsListing = () => {
  return (
    <Content>
      <ModalHeader>
        <LargerBoldText mr="auto">Deployments</LargerBoldText>
      </ModalHeader>

      <Divider sx={{ mb: '1rem' }} />
      <Box sx={{ ml: 'auto', mr: 'auto' }}>
        <DeploymentsTable />
      </Box>
    </Content>
  );
};

export default DeploymentsListing;
