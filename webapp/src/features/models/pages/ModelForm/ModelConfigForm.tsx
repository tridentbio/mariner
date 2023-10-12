import { RefreshSharp } from '@mui/icons-material';
import { Button, InputLabel, TextField } from '@mui/material';
import { Box } from '@mui/system';
import * as modelsApi from 'app/rtk/generated/models';
import IconButton from 'components/atoms/IconButton';
import MDTextField from 'components/organisms/MDTextField';
import ModelAutoComplete from 'features/models/components/ModelAutocomplete';
import { useEffect } from 'react';
import { Control, Controller, useFormContext } from 'react-hook-form';
import { useSearchParams } from 'react-router-dom';
import rehypeSanitize from 'rehype-sanitize';

export interface ModelConfigFormProps {
  control: Control<modelsApi.ModelCreate>;
  onClear?: () => void;
  disabled?: boolean;
}

const ModelConfigForm = ({
  control,
  onClear,
  disabled,
}: ModelConfigFormProps) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const { setValue } = useFormContext();
  const registeredModel = searchParams.get('registeredModel');
  const [getRandomName, { data, isLoading: randomNameLoading }] =
    modelsApi.useLazyGetModelNameSuggestionQuery();

  useEffect(() => {
    if (!data) return;
    if (data) setValue('name', data.name);
  }, [data]);
  return (
    <>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
        }}
        data-testid="model-description-form"
      >
        <Controller
          control={control}
          name="name"
          render={({ field, fieldState }) => (
            <ModelAutoComplete
              {...field}
              data-testid="model-name"
              sx={{ width: '100%' }}
              error={!!fieldState.error}
              label={fieldState.error?.message || 'Model Name'}
              disabled={!!registeredModel || disabled}
              value={{ name: field.value }}
              onChange={(event) => {
                field.onChange({ target: { value: event?.name } });
              }}
              onBlur={field.onBlur}
              id="model-name"
            />
          )}
        />

        {registeredModel && !disabled && (
          <Button variant="text" onClick={() => onClear && onClear()}>
            Clear
          </Button>
        )}
        {!registeredModel && !disabled && (
          <IconButton
            data-testid="random-model-name"
            disabled={randomNameLoading}
            onClick={() => getRandomName()}
          >
            <RefreshSharp />
          </IconButton>
        )}
      </Box>
      <Controller
        control={control}
        name="modelDescription"
        render={({ field, fieldState }) => (
          <TextField
            data-testid="model-description"
            {...field}
            error={!!fieldState.error}
            label={fieldState.error?.message || 'Model Description'}
            sx={{ width: '100%', mt: 1 }}
            disabled={disabled}
          />
        )}
      />
      <Controller
        control={control}
        name="config.name"
        render={({ field, fieldState }) => (
          <TextField
            data-testid="version-name"
            {...field}
            error={!!fieldState.error}
            label={fieldState.error?.message || 'Model Version Name'}
            sx={{ width: '100%', mt: 1 }}
            disabled={disabled}
          />
        )}
      />

      <Box data-color-mode="light">
        <Controller
          name="modelVersionDescription"
          render={({ field, fieldState }) => {
            return (
              <MDTextField
                previewOptions={{ rehypePlugins: [[rehypeSanitize]] }}
                id="description-input"
                data-testid="version-description"
                {...field}
                error={!!fieldState.error}
                label={fieldState.error?.message || 'Model Version Description'}
                onChange={(newValue) =>
                  field.onChange({ target: { value: newValue } })
                }
              />
            );
          }}
        ></Controller>
      </Box>
    </>
  );
};

export default ModelConfigForm;