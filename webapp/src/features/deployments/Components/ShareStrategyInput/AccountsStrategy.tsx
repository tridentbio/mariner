import { Autocomplete, TextField } from '@mui/material';
import { useAppSelector } from 'app/hooks';
import { User } from 'app/rtk/auth';
import React, { useMemo } from 'react';
import { useFormContext } from 'react-hook-form';

type AccountsStrategyProps = {
  usersList: User[];
};

const AccountsStrategy: React.FC<AccountsStrategyProps> = ({ usersList }) => {
  const {
    setValue,
    formState: { errors },
    trigger,
    register,
    watch,
  } = useFormContext();
  const currentDeployment = useAppSelector(
    (state) => state.deployments.current
  );
  const defaultValue = useMemo(() => {
    if (currentDeployment?.usersAllowed) {
      register('usersIdAllowed', { value: currentDeployment.usersIdAllowed });
      return currentDeployment.usersAllowed;
    }
    return [];
  }, []);

  return (
    // Another option is to use a Transfer List
    <Autocomplete
      multiple
      limitTags={5}
      id="tags-outlined"
      options={usersList || []}
      defaultValue={defaultValue}
      getOptionLabel={(option) => option.email}
      filterSelectedOptions
      onChange={(_e, value) => {
        setValue(
          'usersIdAllowed',
          value.map((user) => user.id)
        );
        trigger('usersIdAllowed', { shouldFocus: true });
      }}
      renderInput={(params) => (
        <TextField
          {...params}
          error={!!errors.usersIdAllowed}
          helperText={errors.usersIdAllowed?.message as string}
          label="Users Authorized"
          placeholder={
            watch('usersIdAllowed')?.length ? undefined : 'example@domain.com'
          }
        />
      )}
    />
  );
};

export { AccountsStrategy };
