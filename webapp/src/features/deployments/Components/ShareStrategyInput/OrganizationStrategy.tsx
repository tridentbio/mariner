import {
  Button,
  Chip,
  InputAdornment,
  InputLabel,
  TextField,
} from '@mui/material';
import { Box } from '@mui/system';
import { useNotifications } from 'app/notifications';
import React, { useCallback, useRef } from 'react';
import { Controller, useFormContext } from 'react-hook-form';
import { isValidEmailDomain } from '@utils';

type OrganizationStrategyProps = {};

const OrganizationStrategy: React.FC<OrganizationStrategyProps> = () => {
  const { notifyError } = useNotifications();
  const { control, watch } = useFormContext();
  const inputTextRef = useRef<HTMLDivElement>(null);
  const getInputValue = useCallback(() => {
    const input = inputTextRef.current?.querySelector('input');
    if (!input) return;
    const organizationDomain = `@${input.value}`;
    if (!isValidEmailDomain(organizationDomain)) {
      notifyError('Please enter a valid email domain');
      return;
    }
    input.value = '';
    return organizationDomain;
  }, []);
  const organizationsAllowed: string[] = watch('organizationsAllowed');
  return (
    <Controller
      control={control}
      name="organizationsAllowed"
      render={({ field, fieldState: { error } }) => {
        const handleAddOrganization = () => {
          const organizationDomain = getInputValue();
          if (!organizationDomain) return;
          const organizationAlreadyExists = organizationsAllowed.find(
            (item) => item === organizationDomain
          );
          if (organizationAlreadyExists) {
            notifyError('Organization already included');
            return;
          }
          organizationsAllowed.push(organizationDomain);
          field.onChange(organizationsAllowed);
        };
        const handleDeleteOrganization = (index: number) => {
          organizationsAllowed.splice(index, 1);
          field.onChange(organizationsAllowed);
        };

        return (
          <>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 2,
                mt: 2,
              }}
            >
              <TextField
                ref={inputTextRef}
                sx={{ width: '100%' }}
                label="Organization Email Domain"
                placeholder="example.com"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">@</InputAdornment>
                  ),
                }}
              />
              <Button
                variant="contained"
                color="success"
                size="small"
                onClick={handleAddOrganization}
              >
                Add
              </Button>
            </Box>
            <InputLabel
              sx={{
                mt: 1,
                background: '#fff',
                display: 'inline-block',
                ml: 1,
                px: 0.7,
                fontSize: 16,
                color: 'rgba(0, 0, 0, 0.6)',
                ...(!!error && { color: '#d32f2f' }),
              }}
            >
              Organizations Authorized
            </InputLabel>
            <Box
              sx={{
                border: '1px solid rgb(224, 224, 224)',
                borderRadius: '5px',
                outlineWidth: 1,
                p: 2,
                display: 'flex',
                flexWrap: 'wrap',
                gap: 1,
                minHeight: '80px',
                mt: -2,
                ...(!!error && { borderColor: '#d32f2f' }),
              }}
            >
              {organizationsAllowed?.map((item, index) => (
                <Chip
                  key={item}
                  label={item}
                  onDelete={() => handleDeleteOrganization(index)}
                />
              ))}
            </Box>
            {!!error && (
              <p
                className="MuiFormHelperText-root Mui-error MuiFormHelperText-sizeMedium MuiFormHelperText-contained css-1mnhdd2-MuiFormHelperText-root"
                id="model-version-select-helper-text"
              >
                {error.message}
              </p>
            )}
          </>
        );
      }}
    />
  );
};

export default OrganizationStrategy;
