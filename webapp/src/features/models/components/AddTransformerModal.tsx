import { useGetModelOptionsQuery } from '@app/rtk/generated/models';
import { useMemo, useState } from 'react';
import Modal from 'components/templates/Modal';
import { Box, Button, MenuItem, Select, capitalize } from '@mui/material';

const summarizeFowardArgs = (fowardArgsSummary?: Record<string, any>) =>
  fowardArgsSummary
    ? Object.keys(fowardArgsSummary).reduce(
        (acc, key) => ({
          ...acc,
          [key]: null,
        }),
        {}
      )
    : {
        X: null,
      };

export const AddTransformerModal = ({
  open,
  cancel,
  confirm,
  transfomerType = 'transform',
}: {
  open: boolean;
  cancel: () => void;
  confirm: (transformer: {
    name: string;
    constructorArgs: Record<string, any>;
    fowardArgs: Record<string, null>;
    type: string;
  }) => void;
  transfomerType?: 'transformer' | 'featurizer';
}) => {
  const { data: options } = useGetModelOptionsQuery();
  const [selected, setSelected] = useState<number>(0);

  const allOptions = useMemo(
    () =>
      options
        ?.filter((o) => o.type === transfomerType)
        .map((option) => ({
          name: option.classPath.split('.').at(-1)!,
          constructorArgs: (option.defaultArgs || {}) as Record<string, any>,
          fowardArgs: summarizeFowardArgs(option.component?.fowardArgsSummary),
          type: option.classPath,
        })) || [],
    [options]
  );

  return (
    <Modal
      open={open}
      onClose={cancel}
      title={`Add ${capitalize(transfomerType)}`}
    >
      <Box
        sx={{
          background: 'white',
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '3rem',
          alignItems: 'center',
          margin: 'auto',
        }}
      >
        <Select
          sx={{ width: '60%' }}
          value={selected}
          onChange={({ target: { value } }) => {
            setSelected(value as number);
          }}
        >
          {allOptions.map((transformer, i) => (
            <MenuItem value={i} key={i}>
              {transformer.name}
            </MenuItem>
          ))}
        </Select>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
          <Button variant="outlined" color="warning" onClick={cancel}>
            Cancel
          </Button>
          {
            <Button
              variant="contained"
              color="primary"
              onClick={() => confirm(allOptions[selected])}
            >
              Confirm
            </Button>
          }
        </Box>
      </Box>
    </Modal>
  );
};
