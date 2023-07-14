import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Checkbox,
  FormControlLabel,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import { Controller, useFormContext } from 'react-hook-form';
import MetricSelect from './MetricSelect';
import { required } from 'utils/reactFormRules';
import ModeRadioInput from './ModeRadioInput';
import { MetricMode } from 'app/types/domain/experiments';
import {
  RadioButtonCheckedOutlined,
  RadioButtonUncheckedOutlined,
} from '@mui/icons-material';
import {APITargetConfig} from '@model-compiler/src/interfaces/model-editor'
import { BaseTrainingRequest } from '@app/rtk/generated/experiments';

interface AdvancedOptionsProps {
  open?: boolean;
  onToggle: (opened: boolean) => void;
  targetColumns: APITargetConfig[];
}

const AdvancedOptionsForm = ({
  open,
  onToggle,
  targetColumns,
}: AdvancedOptionsProps) => {
  const { control, setValue, resetField } =
    useFormContext<BaseTrainingRequest>();
  return (
    <Accordion
      elevation={1}
      sx={{ mb: 2 }}
      onChange={(_e, isExpanded) => onToggle(isExpanded)}
    >
      <AccordionSummary
        expandIcon={
          open ? (
            <RadioButtonCheckedOutlined />
          ) : (
            <RadioButtonUncheckedOutlined />
          )
        }
        aria-controls="panel1a-content"
        id="panel1a-header"
        sx={{ backgroundColor: '#e0ecff' }}
      >
        <Typography sx={{ mb: 1 }}>Early Stopping Options</Typography>
      </AccordionSummary>
      <AccordionDetails>
        {open && (
          <>
            <Controller
              rules={{ ...required }}
              control={control}
              defaultValue={'min'}
              name="config.earlyStoppingConfig.metricKey"
              render={({ field, fieldState: { error } }) => (
                <MetricSelect
                  field={field}
                  error={error}
                  setValue={(value: MetricMode) => {
                    setValue('config.earlyStoppingConfig.mode', value);
                  }}
                  targetColumns={targetColumns}
                  reset={() =>
                    resetField('config.earlyStoppingConfig.metricKey')
                  }
                  cleanable
                />
              )}
            />
            <Box>
              <Controller
                rules={{ ...required }}
                control={control}
                name="config.earlyStoppingConfig.minDelta"
                render={({ field, fieldState: { error } }) => (
                  <Tooltip
                    title={
                      'Minimum change in the monitored value required to reset patience'
                    }
                    placement={'right'}
                  >
                    <TextField
                      sx={{ mb: 1 }}
                      aria-label="Minimum Delta"
                      label="Minimum Delta"
                      type="number"
                      inputProps={{ step: '0.01' }}
                      {...field}
                      helperText={error?.message}
                      error={!!error}
                    />
                  </Tooltip>
                )}
              />
            </Box>
            <Box>
              <Controller
                rules={{ ...required }}
                control={control}
                name="config.earlyStoppingConfig.patience"
                render={({ field, fieldState: { error } }) => (
                  <Tooltip
                    title={
                      'Number of epochs to wait for an improvement in the monitored value before stopping the training run'
                    }
                    placement={'right'}
                  >
                    <TextField
                      sx={{ mb: 1 }}
                      aria-label="patience"
                      label="Patience"
                      type="number"
                      {...field}
                      error={!!error}
                      helperText={error?.message}
                    />
                  </Tooltip>
                )}
              />
            </Box>
            <ModeRadioInput fieldName="earlyStoppingConfig.mode" />
            <Tooltip
              title={
                'Halt training if an infinite value is returned for the monitored metric'
              }
              placement={'right'}
            >
              <FormControlLabel
                control={
                  <Controller
                    control={control}
                    defaultValue={false}
                    name="config.earlyStoppingConfig.checkFinite"
                    render={({ field }) => (
                      <Checkbox
                        {...field}
                        checked={field.value}
                        onChange={(e) => field.onChange(e.target.checked)}
                      />
                    )}
                  />
                }
                label="Check Finite"
              />
            </Tooltip>
          </>
        )}
      </AccordionDetails>
    </Accordion>
  );
};

export default AdvancedOptionsForm;
