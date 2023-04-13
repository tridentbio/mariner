import { ErrorSharp } from '@mui/icons-material';
import { Alert } from '@mui/material';
import { Box } from '@mui/system';

interface StackTraceProps {
  message?: string;
  stackTrace?: string;
}
const StackTrace = ({ stackTrace, message }: StackTraceProps) => {
  return stackTrace ? (
    <Box
      sx={{
        mt: 2,
        '& > pre': {
          backgroundColor: 'rgba(0,0,0,0.1)',
          overflowX: 'scroll',
          padding: '8px 4px',
        },
      }}
    >
      <Alert color="error" icon={<ErrorSharp />}>
        {message}
      </Alert>
      <pre>{stackTrace}</pre>
    </Box>
  ) : null;
};

export default StackTrace;
