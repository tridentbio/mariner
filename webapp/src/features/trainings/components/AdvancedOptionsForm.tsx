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
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Controller, useFormContext } from 'react-hook-form';
import MetricSelect from './MetricSelect';
import { required } from 'utils/reactFormRules';
import ModeRadioInput from './ModeRadioInput';
import { TrainingRequest } from 'app/rtk/generated/experiments';
import { MetricMode } from 'app/types/domain/experiments';
import {
  RadioButtonCheckedOutlined,
  RadioButtonUncheckedOutlined,
} from '@mui/icons-material';
import { TargetConfig } from 'app/rtk/generated/models';

interface AdvancedOptionsProps {
  open?: boolean;
  onToggle: (opened: boolean) => void;
  targetColumns: TargetConfig[];
}

const AdvancedOptionsForm = ({
  open,
  onToggle,
  targetColumns,
}: AdvancedOptionsProps) => {
  const { control, setValue, resetField } = useFormContext<TrainingRequest>();
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
              name="earlyStoppingConfig.metricKey"
              render={({ field, fieldState: { error } }) => (
                <MetricSelect
                  field={field}
                  error={error}
                  setValue={(value: MetricMode) => {
                    setValue('earlyStoppingConfig.mode', value);
                  }}
                  targetColumns={targetColumns}
                  reset={() => resetField('earlyStoppingConfig.metricKey')}
                  cleanable
                />
              )}
            />
            <Box>
              <Controller
                rules={{ ...required }}
                control={control}
                name="earlyStoppingConfig.minDelta"
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
                name="earlyStoppingConfig.patience"
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
                    name="earlyStoppingConfig.checkFinite"
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
