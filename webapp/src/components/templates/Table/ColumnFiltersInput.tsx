import { AddOutlined } from '@mui/icons-material';
import { Autocomplete, Box, IconButton, Paper, TextField } from '@mui/material';
import { useState } from 'react';
import { Column, OperatorValue } from './types';

export interface ColumnFiltersInputProps {
  col: Column<any, any>;
  onAddFilter: (
    columnField: string,
    operator: OperatorValue,
    value: any
  ) => any;
}
const inputContainerStyle = {
  sx: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    mb: 2,
  },
} as const;
const FilterInput = ({
  value,
  onChange,
  onDone,
  label,
}: {
  value: string;
  onChange: React.ChangeEventHandler<HTMLInputElement>;
  onDone: () => any;
  label: string;
}) => (
  <Box {...inputContainerStyle}>
    <TextField
      variant="standard"
      label={label}
      value={value}
      onChange={onChange}
    ></TextField>
    <IconButton
      onClick={onDone}
      sx={{
        width: 'fit-content',
        height: 'fit-content',
        padding: 1,
        mt: 'auto',
        mb: 'auto',
      }}
      disabled={!value}
    >
      <AddOutlined />
    </IconButton>
  </Box>
);
const ColumnFiltersInput = ({ col, onAddFilter }: ColumnFiltersInputProps) => {
  const [filterEq, setFilterEq] = useState('');
  const [filterLt, setFilterLt] = useState(0);
  const [filterGt, setFilterGt] = useState(0);
  const [filterIncludes, setFilterIncludes] = useState('');
  const [containsValue, setContainsValue] = useState<any[]>([]);

  return (
    <Box
      component={Paper}
      sx={{ display: 'flex', p: 1, flexDirection: 'column' }}
    >
      {col.filterSchema?.byValue && (
        <FilterInput
          onDone={() => {
            onAddFilter(col.field, 'eq', filterEq);
          }}
          label="Equals"
          value={filterEq}
          onChange={(event) => setFilterEq(event.target.value)}
        />
      )}
      {col.filterSchema?.byIncludes && (
        <FilterInput
          onDone={() => {
            onAddFilter(col.field, 'inc', filterIncludes);
          }}
          label="Includes"
          value={filterIncludes}
          onChange={(event) => setFilterIncludes(event.target.value)}
        />
      )}
      {col.filterSchema?.byLessThan && (
        <FilterInput
          onDone={() => {
            onAddFilter(col.field, 'lt', filterLt);
          }}
          label="Less Than"
          value={filterLt.toString()}
          onChange={(event) => setFilterLt(parseInt(event.target.value))}
        />
      )}
      {col.filterSchema?.byGreaterThan && (
        <FilterInput
          value={filterGt.toString()}
          onChange={(event) => setFilterGt(parseInt(event.target.value))}
          onDone={() => {
            onAddFilter(col.field, 'gt', filterGt);
            setFilterGt(0);
          }}
          label="Greater Than"
        />
      )}

      {col.filterSchema?.byContains && (
        <Box {...inputContainerStyle}>
          <Autocomplete
            multiple
            sx={{ minWidth: 200, maxWidth: 300 }}
            value={containsValue}
            onChange={(_, newValue) => setContainsValue(newValue)}
            getOptionLabel={col.filterSchema?.byContains!.getLabel}
            options={col.filterSchema?.byContains!.options || []}
            renderInput={(input) => <TextField {...input} variant="standard" />}
            disableCloseOnSelect
          />

          <IconButton
            onClick={() =>
              onAddFilter(
                col.field,
                'ct',
                containsValue.map((col) => col.key)
              )
            }
            disabled={!containsValue.length}
            sx={{
              width: 'fit-content',
              height: 'fit-content',
              padding: 1,
              mt: 'auto',
              mb: 'auto',
            }}
          >
            <AddOutlined />
          </IconButton>
        </Box>
      )}
    </Box>
  );
};

export default ColumnFiltersInput;
