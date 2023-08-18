import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Select,
  MenuItem,
  InputLabel,
  Button,
  RadioGroup,
  FormControlLabel,
  Radio,
} from '@mui/material';
import { ControllerRenderProps, FieldError } from 'react-hook-form';
import { Box } from '@mui/system';
import {
  GetExperimentsMetricsApiResponse,
  useGetExperimentsMetricsQuery,
} from 'app/rtk/generated/experiments';
import { BaseTrainingRequest } from '@app/types/domain/experiments';
// TODO: fix MathJax in TexMath
// import TexMath from 'components/atoms/TexMath';
import { defaultModeIsMax } from 'utils';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { ModelVersionType } from 'app/types/domain/models';
import { APITargetConfig } from '@model-compiler/src/interfaces/model-editor';

type MetricSelectProps = {
  field: ControllerRenderProps<BaseTrainingRequest, any>;
  error?: FieldError;
  setValue: (value: 'min' | 'max') => void;
  reset?: () => void;
  cleanable?: boolean;
  targetColumns: APITargetConfig[];
};

const sortFilterMetrics = (
  _metrics: GetExperimentsMetricsApiResponse | undefined,
  column: APITargetConfig | null
) => {
  const modelVersionType = {
    binary: 'classification',
    multiclass: 'classification',
    regression: 'regressor',
  }[column?.columnType || 'regression'] as ModelVersionType;
  return [...(_metrics || [])]
    .sort((a, b) => (a.key.toLowerCase() < b.key.toLowerCase() ? -1 : 1))
    .filter((metric) => metric.type === modelVersionType);
};

const MetricSelect: React.FC<MetricSelectProps> = ({
  error,
  field: { onChange, name, ref },
  setValue,
  targetColumns,
  reset = () => {},
  cleanable = false,
}) => {
  const { data: metrics } = useGetExperimentsMetricsQuery();
  const [stage, setStage] = useState<'train' | 'val'>('val');
  const [selected, setSelected] = useState<string | null>('');
  const [column, setColumn] = useState<APITargetConfig | null>(null);

  useEffect(() => {
    // simulation of event property
    if (selected && column && stage)
      onChange({
        target: {
          value: `${stage}/${selected}/${column.name}`,
        },
      });
  }, [stage, selected, column]);

  const getColumn = useCallback(
    (name: string) =>
      targetColumns.find((column) => column.name === name) || null,
    [targetColumns]
  );

  const sortedFilteredMetrics = useMemo(
    () => sortFilterMetrics(metrics, column),
    [metrics, column]
  );

  return (
    <Box sx={{ mb: 1, width: '100%' }}>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '1rem',
          width: '100%',
        }}
      >
        <>
          <Box sx={{ width: 'inherit' }}>
            <InputLabel>Target Column</InputLabel>
            <Select
              disabled={!targetColumns?.length}
              sx={{ width: '100%' }}
              onChange={(event) => {
                setColumn(getColumn(event.target.value as string));
                setValue(
                  defaultModeIsMax(event.target.value as string) ? 'max' : 'min'
                );
              }}
              value={column?.name || ''}
              ref={ref}
              label={'Target Column'}
            >
              {targetColumns.map((column) => (
                <MenuItem key={column.name} value={column.name}>
                  {column.name}
                </MenuItem>
              ))}
            </Select>
          </Box>
          <Box sx={{ width: 'inherit' }}>
            <InputLabel>Metric to monitor</InputLabel>
            <Select
              sx={{ width: '100%' }}
              onChange={(event) => {
                setSelected(event.target.value);
                setValue(defaultModeIsMax(event.target.value) ? 'max' : 'min');
              }}
              value={selected || ''}
              ref={ref}
              name={name}
              label={error?.message || undefined}
            >
              {sortedFilteredMetrics.map((metric) => (
                <MenuItem key={metric.key} value={metric.key}>
                  {metric.texLabel?.replace('^2', '²') || metric.label}
                </MenuItem>
              ))}
            </Select>
          </Box>
          <RadioGroup
            row
            value={stage}
            onChange={(event) =>
              setStage(event.target.value as 'train' | 'val')
            }
          >
            <FormControlLabel
              value="train"
              label="Train"
              sx={{ m: 0 }}
              control={<Radio />}
            />
            <FormControlLabel
              value="val"
              label="Validation"
              sx={{ m: 0 }}
              control={<Radio />}
            />
          </RadioGroup>

          {cleanable && (
            <Button
              onClick={() => {
                reset();
                setSelected('');
              }}
            >
              <DeleteOutlineIcon sx={{ color: 'grey.500' }} />
            </Button>
          )}
        </>
      </Box>
    </Box>
  );
};

export default MetricSelect;
