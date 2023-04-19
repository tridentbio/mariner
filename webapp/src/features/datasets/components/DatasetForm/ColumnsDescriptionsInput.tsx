import { ColumnInfo, DataTypeDomainKind } from 'app/types/domain/datasets';
import { Text } from 'components/molecules/Text';
import { Box } from '@mui/system';
import {
  Chip,
  TextField,
  Autocomplete,
  Alert,
  IconButton,
  Tooltip,
  Divider,
  Badge,
} from '@mui/material';
import DataTypeInput from './inputs/DataTypeInput';
import { testNonePattern, testPattern } from 'utils';
import {
  AddOutlined,
  ErrorSharp,
  RemoveSharp,
  WarningOutlined,
} from '@mui/icons-material';
import {
  Control,
  Controller,
  useFieldArray,
  useFormContext,
} from 'react-hook-form';
import { required } from 'utils/reactFormRules';
import { DatasetForm } from './types';

interface ColumnsDescriptionsInputProps {
  columnsMeta: ColumnInfo[];
  label?: string;
  error?: boolean;
  control: Control<DatasetForm, any>;
}

export const ColumnsDescriptionsInput = ({
  control,
  columnsMeta,
  error,
  label,
}: ColumnsDescriptionsInputProps) => {
  const { watch, getValues, register } = useFormContext<DatasetForm>();
  const { append, remove, fields } = useFieldArray({
    control,
    name: 'columnsMetadata',
  });
  const columnsMetadata = watch('columnsMetadata');
  const testMatchSome = (pattern: string) =>
    columnsMeta.map((col) => col.name).some((val) => testPattern(val)(pattern));

  const validatePattern = (pattern: string) => {
    if (
      columnsMetadata &&
      columnsMetadata.map((cd) => cd.pattern).filter((p) => p === pattern)
        .length > 1
    ) {
      return 'Pattern already described';
    } else if (!checkRegex(pattern)) {
      return 'Invalid regex';
    } else if (!testMatchSome(pattern)) {
      return "Doesn't match any column";
    }
  };

  const checkRegex = (patternStr: string) => {
    try {
      new RegExp(patternStr);
      return true;
    } catch (err) {
      return false;
    }
  };

  const undescribedColumns = columnsMetadata
    ? columnsMeta
        .map((col) => col.name)
        .filter(testNonePattern(columnsMetadata.map((col) => col.pattern)))
    : [];

  const handleDelete = (index: number) => {
    remove(index);
  };
  return (
    <div>
      <Text fontSize={20} fontWeight="bold" mb={1}>
        Columns Descriptions:
      </Text>
      {error && (
        <Alert color="error" icon={<ErrorSharp />}>
          {label}
        </Alert>
      )}
      <Box sx={{ mb: 1 }}>
        {!!columnsMeta.length && (
          <Box
            sx={{
              lineHeight: 3,
              maxHeight: 250,
              overflowY: 'auto',
              pt: 2,
              pb: 2,
              pr: 2,
              pl: 2,
            }}
          >
            {columnsMeta.map((col) => {
              const chip = (
                <Chip
                  sx={{ mr: 1 }}
                  key={col.name}
                  color={
                    undescribedColumns.includes(col.name)
                      ? 'default'
                      : 'primary'
                  }
                  label={col.name}
                />
              );
              return col.name.startsWith('Unnamed: ') ? (
                <Badge
                  key={col.name + 'badge'}
                  title="Unnamed column from uploaded file"
                  color="warning"
                  badgeContent={
                    <WarningOutlined sx={{ width: 10, height: 10 }} />
                  }
                  anchorOrigin={{
                    vertical: 'top',
                    horizontal: 'left',
                  }}
                >
                  {chip}
                </Badge>
              ) : (
                chip
              );
            })}
          </Box>
        )}
      </Box>
      <Box>
        <Text>
          Use a valid regular expression to describe a set of related column
          names.
        </Text>
        {fields.map(({ pattern, id }, i) => (
          <div key={id}>
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                mb: 2,
                mt: 2,
              }}
            >
              <Tooltip title="Remove this column description">
                <IconButton onClick={() => handleDelete(i)}>
                  <RemoveSharp />
                </IconButton>
              </Tooltip>
              <Box
                data-testid={`input-group-${pattern}`}
                sx={{
                  width: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                <Controller
                  control={control}
                  name={`columnsMetadata.${i}.pattern`}
                  rules={{
                    ...required,
                    validate: validatePattern,
                  }}
                  render={({ field, fieldState: { error } }) => {
                    const currentValue = {
                      value: field.value,
                      raw: false,
                      label: field.value,
                    };
                    return (
                      <Autocomplete
                        sx={{ mt: 1 }}
                        selectOnFocus
                        clearOnBlur
                        handleHomeEndKeys
                        options={undescribedColumns
                          .map((label) => ({ label, value: label, raw: false }))
                          .concat(
                            pattern === '' ||
                              undescribedColumns.some((col) =>
                                col.includes(pattern)
                              )
                              ? []
                              : [currentValue]
                          )}
                        onChange={(_event, option) => {
                          field.onChange({
                            target: { value: option?.value || '' },
                          });
                        }}
                        value={currentValue}
                        noOptionsText="No columns"
                        renderInput={(params) => (
                          <TextField
                            error={!!error}
                            helperText={error?.message}
                            label={'Column Name Pattern'}
                            placeholder="col_pattern_*"
                            onChange={(event) => {
                              field.onChange(event.target.value);
                            }}
                            {...params}
                            data-testid="dataset-col-name-input"
                          />
                        )}
                      />
                    );
                  }}
                />

                <TextField
                  data-testid="dataset-col-description-input"
                  defaultValue={getValues(`columnsMetadata.${i}.description`)}
                  {...register(`columnsMetadata.${i}.description`)}
                  data-descriptionPattern={`${pattern}`}
                  sx={{ mt: 1 }}
                  multiline
                  label={`Description ${i + 1}`}
                  type="standard"
                />

                <Box sx={{ width: '100%', mt: 1 }}>
                  <DataTypeInput
                    pattern={pattern}
                    label={`Data Type ${i + 1}`}
                    index={i}
                    control={control}
                  />
                </Box>
              </Box>
            </Box>
            <Divider />
          </div>
        ))}
      </Box>

      {!!undescribedColumns.length && fields.length < columnsMeta.length && (
        <IconButton
          title="Add a column description"
          color="primary"
          onClick={() =>
            append({
              pattern: '',
              description: '',
              dataType: {
                domainKind: DataTypeDomainKind.String,
              },
            })
          }
          id="add-col-description"
        >
          <AddOutlined />
        </IconButton>
      )}
    </div>
  );
};
