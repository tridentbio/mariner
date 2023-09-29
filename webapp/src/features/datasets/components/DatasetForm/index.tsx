import {
  TextField,
  Box,
  MenuItem,
  InputLabel,
  Container,
  Button,
  CircularProgress,
  Autocomplete,
  FormHelperText,
} from '@mui/material';
import { lazy, ReactNode, useState } from 'react';
import { ColumnsDescriptionsInput } from './ColumnsDescriptionsInput';
import { useNotifications } from '../../../../app/notifications';
import { sum, testNonePattern } from 'utils';
import * as Datasets from 'app/rtk/generated/datasets';
import { datasetsApi as dsApi } from 'app/rtk/datasets';
import Loading from 'components/molecules/Loading';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import { required } from 'utils/reactFormRules';
import { ColumnInfo, SplitType } from 'app/types/domain/datasets';
import { DatasetForm as IDatasetForm } from './types';
const { useGetColumnsMetadataMutation } = dsApi;

const MDTextField = lazy(() => import('components/organisms/MDTextField'));

const itemLayout = {
  sx: {
    mb: 1,
  },
} as const;

interface DatasetFormProps {
  initialValues?: Datasets.Dataset;
  onSubmit?: (values: Datasets.BodyCreateDatasetApiV1DatasetsPost) => void;

  onCancel?: () => void;
  loading?: boolean;
}

const getDefaultFormValues = (
  dataset: DatasetFormProps['initialValues']
): IDatasetForm => {
  if (!dataset) {
    // resolver: ajvResolver(datasetsApi.CreateDatasetApiArgSchema, {}),
    return {
      name: '',
      description: '',
      splitType: SplitType.random,
      splitTarget: '',
      columnsMetadata: [],
    };
  } else {
    return dataset;
  }
};

const DatasetForm = ({ initialValues, ...props }: DatasetFormProps) => {
  const methods = useForm<IDatasetForm>({
    defaultValues: getDefaultFormValues(initialValues),
  });
  const {
    handleSubmit: onSubmit,
    getValues,
    reset,
    watch,
    control,
    formState: { errors },
  } = methods;
  const splitType = watch('splitType');
  const [getColumnsMetadata, { isLoading: metaLoading }] =
    useGetColumnsMetadataMutation();

  const [columnsMeta, setColumnsMeta] = useState<ColumnInfo[] | undefined>(
    initialValues?.columnsMetadata?.map(
      ({ pattern, dataType, description }) => ({
        name: pattern,
        dtype: dataType,
        description,
      })
    )
  );
  const { setMessage } = useNotifications();

  const handleSubmit = onSubmit(
    (data) => {
      if (props.onSubmit) {
        return props.onSubmit({
          columnsMetadata: JSON.stringify(data.columnsMetadata),
          name: data.name,
          description: data.description,
          splitType: data.splitType,
          splitTarget: data.splitTarget,
          file: data.file,
          splitOn: data.splitOn,
        });
      }
    },
    (msg) => {
      setMessage({ message: 'Failed to submit dataset', type: 'error' });
    }
  );

  const getSomeRowsCSV = async (
    file: File,
    nRows: number = 20
  ): Promise<File> => {
    const reader = new FileReader();
    reader.readAsText(file);

    return new Promise((resolve, reject) => {
      reader.onload = async (event) => {
        const text = ((event.target?.result as string) || '')
          .split('\n', nRows)
          .join('\n');

        const newFile = new File([text as string], file.name || 'file.csv', {
          type: 'text/plain',
        });
        resolve(newFile);
      };

      reader.onerror = (error) => reject(error);
    });
  };

  const fetchCSVData = async (file: File | Blob) => {
    const MAX_SIZE = 1000000;
    try {
      const data = await getColumnsMetadata(
        file.size > MAX_SIZE ? await getSomeRowsCSV(file as File) : file
      ).then((result) => {
        if ('data' in result && result.data) return result.data;
        throw new Error('Failed to get csv columns');
      });

      setColumnsMeta(data);
      reset(
        {
          ...getValues(),
          columnsMetadata: data
            .filter((col) => col.dtype)
            .map((col) => ({
              pattern: `${col.name}`,
              dataType: col.dtype,
              description: '',
            })),
        },
        { keepErrors: true, keepDirty: true }
      );
    } catch (err) {
      setMessage({ type: 'error', message: 'Failed to get csv columns' });
    }
  };

  return (
    <FormProvider {...methods}>
      <form autoComplete="off" onSubmit={handleSubmit}>
        <Box sx={{ m: 1, display: 'flex', flexDirection: 'column' }}>
          <Controller
            rules={{
              ...required,
            }}
            shouldUnregister
            name="name"
            control={control}
            render={({ field, fieldState: { error } }) => (
              <TextField
                label={'Name'}
                helperText={error?.message}
                id="dataset-name-input"
                error={!!errors.name}
                {...itemLayout}
                {...field}
              />
            )}
          />
          <Controller
            name="description"
            rules={{
              ...required,
            }}
            control={control}
            render={({ field, fieldState: { error } }) => (
              <MDTextField
                error={!!error}
                {...field}
                onChange={(mdStr) =>
                  field.onChange({ target: { value: mdStr } })
                }
                label="Description"
                helperText={error?.message}
              />
            )}
          />
          {!initialValues && (
            <Box sx={{ mb: 1.5 }}>
              <Controller
                rules={{
                  ...required,
                }}
                control={control}
                name="file"
                render={({ field, fieldState: { error } }) => (
                  <>
                    <InputLabel htmlFor="dataset-upload" error={!!error}>
                      File
                    </InputLabel>
                    <input
                      type="file"
                      id="dataset-upload"
                      onChange={(event) => {
                        if (event.target.files && event.target.files.length) {
                          field.onChange({
                            target: { value: event.target.files[0] },
                          });
                          fetchCSVData(event.target.files[0]);
                        }
                      }}
                      accept=".csv"
                    />
                  </>
                )}
              />
              {!!errors.file && (
                <FormHelperText error>
                  {errors.file?.message || ''}
                </FormHelperText>
              )}
            </Box>
          )}
          <Loading
            isLoading={metaLoading}
            message="Wait while we check the data types of your dataset"
          />
          {columnsMeta && (
            <>
              <Controller
                rules={{ ...required }}
                name="splitType"
                render={({ field }) => (
                  <TextField
                    disabled={!!initialValues}
                    label="Split Type"
                    sx={{ mb: 1 }}
                    id="dataset-splittype-input"
                    helperText={
                      errors.splitType?.message ||
                      ('Stratagy used to split the dataset' as ReactNode)
                    }
                    select
                    error={!!errors.splitType}
                    {...field}
                  >
                    <MenuItem value={SplitType.random}>Random</MenuItem>
                    {columnsMeta.some(
                      (column) => column.dtype?.domainKind === 'smiles'
                    ) && (
                      <MenuItem value={SplitType.scaffold}>Scaffold</MenuItem>
                    )}
                  </TextField>
                )}
                control={control}
              />
              {splitType === 'scaffold' ? (
                <Controller
                  control={control}
                  rules={{ ...required }}
                  name="splitOn"
                  render={({ field }) => (
                    <Autocomplete<ColumnInfo>
                      onChange={(_, column) =>
                        field.onChange(column?.name || '')
                      }
                      onBlur={field.onBlur}
                      ref={field.ref}
                      id="dataset-split-column-input"
                      renderInput={(inputProps) => (
                        <TextField
                          label={'Split on'}
                          helperText={errors.splitOn?.message || ''}
                          error={!!errors.splitOn}
                          {...itemLayout}
                          {...inputProps}
                          disabled={!!initialValues}
                        />
                      )}
                      value={columnsMeta.find(
                        (col) => col.name === field.value
                      )}
                      getOptionLabel={(option) => option.name}
                      options={columnsMeta.filter(
                        (column) => column.dtype?.domainKind === 'smiles'
                      )}
                    />
                  )}
                />
              ) : null}
              <Controller
                name="splitTarget"
                shouldUnregister
                control={control}
                rules={{
                  ...required,
                  validate: {
                    value: (val: string) => {
                      if (typeof val === 'string') {
                        const parts = val.split('-');
                        if (parts.length !== 3)
                          return 'Should be a valid split division, e.g.: 60-20-20, 80-10-10';
                        const intParst = parts.map((s) => parseInt(s));
                        if (intParst.includes(NaN))
                          return 'Should only have ints';
                        if (sum(intParst) !== 100)
                          return 'Should sum up to 100';
                      }
                    },
                  },
                }}
                render={({ field, fieldState: { error } }) => (
                  <TextField
                    label={'Split'}
                    helperText={
                      error?.message || 'Training-Testing-Validation %s'
                    }
                    error={!!error}
                    placeholder="60-20-20"
                    id="dataset-split-input"
                    {...itemLayout}
                    {...field}
                    disabled={!!initialValues}
                  />
                )}
              />
              <Controller
                control={control}
                name="columnsMetadata"
                rules={{
                  validate: (vals) => {
                    const undescribed = columnsMeta.filter((colMeta) => {
                      return testNonePattern(
                        (vals || []).map((val) => val.pattern)
                      )(colMeta.name);
                    });
                    if (undescribed.length) {
                      return (
                        'Some columns are not described: ' +
                        undescribed.map((c) => c.name).join(', ')
                      );
                    }
                  },
                }}
                render={({ fieldState: { error } }) => (
                  <>
                    <ColumnsDescriptionsInput
                      error={!!error?.root}
                      label={error?.root?.message}
                      columnsMeta={columnsMeta}
                      control={control}
                    />
                  </>
                )}
              />
            </>
          )}
          <Container
            sx={{
              display: 'flex',
              flexDirection: 'row',
              justifyContent: 'flex-end',
              mt: 3,
            }}
          >
            <Button onClick={props.onCancel}>Close</Button>
            <Box sx={{ position: 'relative', ml: 5 }}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={props.loading}
                id="save"
              >
                Save
                {props.loading && (
                  <CircularProgress size={30} sx={{ position: 'absolute' }} />
                )}
              </Button>
            </Box>
          </Container>
        </Box>
      </form>
    </FormProvider>
  );
};

export default DatasetForm;
