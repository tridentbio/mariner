import {
  Autocomplete,
  Chip,
  CircularProgress,
  MenuItem,
  TextField,
} from '@mui/material';
import { useAppDispatch, useAppSelector } from '@hooks';
import { debounce } from 'utils';
import { isUnitValid, Unit } from 'features/units/unitsAPI';
import { fetchUnits } from 'features/units/unitsSlice';
import { useEffect, useMemo, useState } from 'react';
import { useNotifications } from 'app/notifications';

interface UnitsAutocompleteProps {
  value?: Unit;
  onChange?: (unit: Unit | null) => any;
  label?: string;
  error?: boolean | string;
  pattern?: string;
}

const ValidUnit = () => (
  <Chip sx={{ ml: 1 }} color="primary" label="valid unit" />
);
const InvalidUnit = () => (
  <Chip sx={{ ml: 1 }} color="error" label="invalid unit" />
);

type UnitOption = Unit & { type?: 'op' | 'unit' | 'raw' };

const UnitAutocomplete = ({
  value,
  label,
  onChange,
  error,
  pattern,
}: UnitsAutocompleteProps) => {
  const options = useAppSelector((state) => state.units.units);
  const dispatch = useAppDispatch();
  const [validUnit, setValidUnit] = useState(false);
  const [checkLoading, setCheckLoading] = useState(false);
  const { notifyError } = useNotifications();

  useEffect(() => {
    dispatch(fetchUnits());
  }, []);

  const debouncedCheckValid = debounce((unit: string) => {
    if (!unit) {
      setValidUnit(false);
      return;
    }
    setCheckLoading(true);
    isUnitValid(unit)
      .then(setValidUnit)
      .catch(() => {
        setValidUnit(false);
      })
      .finally(() => setCheckLoading(false));
  }, 500);
  const optionsWithValue = useMemo(
    () =>
      value && options.map((opt) => opt.name).includes(value.name)
        ? options
        : [{ name: value?.name, type: 'raw' }, ...options],
    [options, value]
  );
  return (
    <Autocomplete
      sx={{ mt: 2, width: '100%' }}
      data-testid={`dataset-col-data-type-unit-${pattern}`}
      renderInput={(params) => (
        <TextField
          placeholder="mole, meters, mole/litre, ..."
          label={'Unit'}
          helperText={error || ''}
          error={!!error}
          {...params}
        />
      )}
      filterOptions={(options, state) => {
        options.sort((a, b) => (a.name < b.name ? 1 : -1));
        return [
          ...(state.inputValue &&
          !options.map((opt) => opt.name).includes(state.inputValue)
            ? [{ name: state.inputValue, type: 'raw' as const }]
            : []),
          ...options
            .filter(
              (opt) =>
                opt.type === 'raw' ||
                opt.name.toLowerCase().includes(state.inputValue.toLowerCase())
            )
            .slice(0, 100),
        ];
      }}
      renderOption={(props, option) => {
        return (
          <MenuItem {...props}>
            {option.name}
            {option.name && (
              <>
                {option.type === 'raw' && checkLoading && (
                  <CircularProgress sx={{ ml: 1 }} />
                )}
                {option.type === 'raw' && !checkLoading && validUnit && (
                  <ValidUnit />
                )}
                {option.type === 'raw' && !checkLoading && !validUnit && (
                  <InvalidUnit />
                )}
              </>
            )}
          </MenuItem>
        );
      }}
      getOptionLabel={(option) => option?.name || ''}
      options={optionsWithValue as UnitOption[]}
      value={value as UnitOption}
      onChange={(_event, newValue) => {
        if (newValue?.type === 'raw') {
          if (checkLoading) {
            notifyError('Wait while we validate the unit');
            return;
          } else if (!validUnit) {
            notifyError(
              'Unit is not parseble, compose valid units like "mol/litre", "volts*hour"'
            );
            return;
          }
          onChange && onChange(newValue);
        } else {
          onChange && onChange(newValue);
        }
      }}
      onInputChange={(_event, input) => {
        debouncedCheckValid(input);
      }}
      isOptionEqualToValue={(a, b) => a?.name === b?.name}
    />
  );
};

export default UnitAutocomplete;
