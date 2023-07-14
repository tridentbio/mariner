import { Control } from 'react-hook-form';
import { Transformer } from '../pages/ModelCreateV2/DatasetConfigurationForm';
import { ModelCreate } from '@app/rtk/generated/models';
import { Box, Typography } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';

type TransformerConstructorFormProps = {
  control: Control<ModelCreate>;
  transformer: Transformer;
};

const redableTransformerName = (transformer: Transformer) =>
  transformer.name.split('-')[1];

export const TransformerConstructorForm = ({
  control,
  transformer,
}: TransformerConstructorFormProps) => {
  return (
    <Box
      sx={{
        background: 'white',
        padding: '1rem',
        borderRadius: '1rem',
        boxShadow: '0px 4px 4px rgba(0, 0, 0, 0.25)',
        marginBottom: '1rem',
        minHeight: '12rem',
        maxWidth: '60%',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem',
        }}
      >
        <Typography>{redableTransformerName(transformer)}</Typography>
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
  );
};
