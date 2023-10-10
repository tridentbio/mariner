import { CircularProgress, Typography } from '@mui/material';
import { BoxProps, Box } from '@mui/system';
import { useLayoutEffect, useState } from 'react';

interface LoadingProps extends BoxProps {
  isLoading: boolean;
  message?: string;
}
const Loading = ({ isLoading, message, ...boxProps }: LoadingProps) => {
  const [dots, setDots] = useState('');
  useLayoutEffect(() => {
    if (isLoading && message) {
      const interval = setInterval(() => {
        setDots((dots) => (dots === '...' ? '.' : dots + '.'));
      }, 600);
      return () => {
        clearInterval(interval);
      };
    }
  }, [isLoading]);
  return isLoading ? (
    <Box
      sx={{
        display: 'flex',
        padding: 2,
        flexDirection: 'row',
        alignItems: 'center',
      }}
      {...boxProps}
    >
      <CircularProgress size={30} />
      <Typography marginLeft={3} width="80%">
        {message}
        {dots}
      </Typography>
    </Box>
  ) : null;
};

export default Loading;
