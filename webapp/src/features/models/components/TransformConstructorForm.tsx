import { Transformer } from '../pages/ModelCreateV2/DatasetConfigurationForm';
import { Box, Typography } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';

type TransformerConstructorFormProps = {
  transformer: Transformer;
};

const redableTransformerName = (transformer: Transformer) =>
  transformer.name.split('-')[1];

export const TransformConstructorForm = ({
  transformer,
}: TransformerConstructorFormProps) => {
  return (
    <Box
      sx={{
        background: 'white',
        padding: '1rem',
        borderRadius: '1rem',
        border: '1px solid #E0E0E0',
        marginBottom: '1rem',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem',
          padding: '1rem',
        }}
      >
        <Typography>{redableTransformerName(transformer)}</Typography>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-evenly',
            alignItems: 'center',
          }}
        >
          <DeleteIcon />
          <ArrowUpwardIcon />
          <ArrowDownwardIcon />
        </Box>
      </Box>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem',
        }}
      >
        {Object.entries(transformer.constructorArgs).map(
          ([arg, defaultValue]) => (
            <Box
              key={arg}
              sx={{
                display: 'flex',
                justifyContent: 'space-evenly',
                alignItems: 'center',
              }}
            >
              <Typography>{arg}</Typography>
              <Typography>{typeof defaultValue}</Typography>
            </Box>
          )
        )}
      </Box>
    </Box>
  );
};
