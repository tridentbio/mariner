import {
  Alert,
  capitalize,
  FormControlLabel,
  Radio,
  RadioGroup,
} from '@mui/material';
import { usersApiRtk } from 'app/rtk/users';
import React, { useMemo } from 'react';
import { ControllerRenderProps, useFormContext } from 'react-hook-form';
import { DeploymentFormFields, EShareStrategies } from '../../types';
import { AccountsStrategy } from './AccountsStrategy';
import OrganizationStrategy from './OrganizationStrategy';

type ShareStrategyInputProps = {
  field: ControllerRenderProps<DeploymentFormFields, 'shareStrategy'>;
};
const ShareStrategyInput: React.FC<ShareStrategyInputProps> = ({ field }) => {
  const { setValue, unregister } = useFormContext();
  const { data: usersList } = usersApiRtk.useGetUsersQuery({}, {});

  const allowedAccessInputMap = useMemo(
    () => ({
      [EShareStrategies.PUBLIC]: (
        <Alert variant="outlined" severity="error" sx={{ fontSize: 14 }}>
          You have selected to make this model public. Anyone with the URL will
          be able to access and use this model without logging in.
        </Alert>
      ),
      [EShareStrategies.PRIVATE]: (
        <>
          <AccountsStrategy usersList={usersList || []} />
          <OrganizationStrategy />
        </>
      ),
    }),
    [usersList]
  );

  return (
    <>
      <RadioGroup
        row
        aria-labelledby="shareStrategy-radio-buttons-group"
        {...field}
        onChange={(e) => {
          field.onChange(e);
          setValue('usersIdAllowed', []);
          setValue('organizationsAllowed', []);
          unregister('usersIdAllowed', { keepValue: false });
          unregister('organizationsAllowed');
        }}
      >
        <FormControlLabel
          label={capitalize(EShareStrategies.PRIVATE)}
          value={EShareStrategies.PRIVATE}
          control={<Radio />}
        />
        <FormControlLabel
          label={capitalize(EShareStrategies.PUBLIC)}
          value={EShareStrategies.PUBLIC}
          control={<Radio />}
        />
      </RadioGroup>

      {allowedAccessInputMap[field.value]}
    </>
  );
};

export default ShareStrategyInput;
