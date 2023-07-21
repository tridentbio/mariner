import api from '@app/api';
import { Typography } from '@mui/material';
import { Box, useTheme } from '@mui/system';
import { useEffect, useState } from 'react';

interface ApiMetadata {
  name: string;
  version: string;
  description: string;
  tenant: {
    name: string;
  };
}

const getApiMetadata = (): Promise<ApiMetadata> =>
  api.get('/metadata').then((res) => res.data);

const Footer = () => {
  const [metadata, setMetadata] = useState<ApiMetadata | null>(null);
  const theme = useTheme();
  useEffect(() => {
    getApiMetadata().then((data) => setMetadata(data));
  }, []);

  return (
    <Box
      sx={{
        color: 'white',
        padding: 2,
        backgroundColor: theme.palette.primary.main,
      }}
    >
      {metadata && (
        <>
          <Typography fontSize="small">Version {metadata.version}</Typography>
        </>
      )}
    </Box>
  );
};

export default Footer;
