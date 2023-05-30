import { DeploymentWithTrainingData } from '@app/rtk/generated/deployments';
import { LargerBoldText } from '@components/molecules/Text';
import ModalHeader from '@components/templates/Modal/ModalHeader';
import { Box, Divider, Typography } from '@mui/material';
import StatusChip from './StatutsChip';

export const DeploymentHeader = ({
  deployment,
}: {
  deployment: DeploymentWithTrainingData;
}) => (
  <>
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <ModalHeader>
        <LargerBoldText mr="auto">{deployment.name}</LargerBoldText>
      </ModalHeader>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
          alignItems: 'flex-end',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <StatusChip
            sx={{
              textTransform: 'uppercase',
              fontWeight: 'bold',
              padding: '0.5rem',
            }}
            status={deployment.status}
          />
        </Box>
        <Box
          sx={{
            mb: '1rem',
            display: 'flex',
            flexDirection: 'row',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <Typography>
            Rate Limit:{' '}
            <b>{`${deployment.predictionRateLimitValue}/${deployment.predictionRateLimitUnit}`}</b>
          </Typography>
        </Box>
      </Box>
    </Box>

    <Divider sx={{ mb: '1rem' }} />
  </>
);
