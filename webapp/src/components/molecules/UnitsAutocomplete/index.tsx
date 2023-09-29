import { VirtualizedAutocomplete } from '@components/atoms/VirtualizedAutocomplete';
import {
  Chip,
  CircularProgress,
  MenuItem,
  TextField
} from '@mui/material';
import { useAppDispatch, useAppSelector } from 'app/hooks';
import { useNotifications } from 'app/notifications';
import { Unit, isUnitValid } from 'features/units/unitsAPI';
import { fetchUnits } from 'features/units/unitsSlice';
import { useEffect, useMemo, useState } from 'react';
import { debounce } from 'utils';

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
    () => {
      if(!!value?.name && !options.some(opt => opt.name === value.name))
        return [{ name: value?.name, type: 'raw' }, ...options]

      return options
    }, [options, value]);

  const filterOptions = (options: UnitOption[], inputValue: string) => {
    options = options
      .sort((a, b) => a.name.localeCompare(b.name))
      .filter(opt =>
        opt.type === 'raw' ||
        opt.name.toLowerCase().includes(inputValue.toLowerCase())
      );

    const invalidOption = !!inputValue && !options.some((opt) => opt.name == inputValue)

    if(invalidOption)
      options.unshift({ name: inputValue, type: 'raw' as const })
    
    return options
  }

  return (
    <VirtualizedAutocomplete
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
      filterOptions={(options, state) => filterOptions(options, state.inputValue)}
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
