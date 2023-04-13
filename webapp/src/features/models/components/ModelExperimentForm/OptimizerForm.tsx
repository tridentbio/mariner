import { MenuItem, TextFieldProps, Box, TextField } from '@mui/material';
import {
  useGetTrainingExperimentOptimizersQuery,
  TrainingRequest,
  GetTrainingExperimentOptimizersApiResponse,
} from 'app/rtk/generated/experiments';

interface OptimizerProps extends Omit<TextFieldProps, 'value' | 'onChange'> {
  onChange: (optimizerConfig: TrainingRequest['optimizer']) => void;
  value: TrainingRequest['optimizer'];
  helperText?: string;
}

const findOption = (
  options: GetTrainingExperimentOptimizersApiResponse | undefined,
  classPath: string
) => {
  if (!options) return;
  const option = options.find((optim) => optim.classPath === classPath);
  return option;
};
/**
 *  Component that renders inputs for optimizers based on each optimizer input
 *  schema, gotten through the API.
 *
 * @param {OptimizerProps} props - [TODO:description]
 */
const OptimizerForm = ({
  name,
  value,
  onChange,
  helperText,
  ...selectProps
}: OptimizerProps) => {
  const { data: optimizerOptions, isLoading } =
    useGetTrainingExperimentOptimizersQuery();
  if (isLoading) return null;

  const optionParams =
    optimizerOptions &&
    optimizerOptions.find(
      (optimizer) => optimizer.classPath === value?.classPath
    );

  const handleClassPathChange = (event: { target: { value: string } }) => {
    const option = findOption(optimizerOptions, event.target.value as string);
    if (!option) return;
    const newOptimizer = { classPath: event.target.value, params: {} };
    Object.entries(option).forEach(([key, value]) => {
      if (key === 'classPath') return;
      // @ts-ignore
      newOptimizer.params[key] = value?.default;
    });
    // @ts-ignore
    onChange(newOptimizer);
  };
  const handleArgChange = (
    event: { target: { value: string } },
    key: string
  ) => {
    if (!value) return;
    const option = findOption(optimizerOptions, value?.classPath);
    if (!option) return;
    const newOptimizer = { ...value };
    // @ts-ignore
    // if (option.paramType.startsWith('float'))
    if (option[key].paramType.startsWith('float'))
      // @ts-ignore
      newOptimizer.params[key] = parseFloat(event.target.value);
    onChange(newOptimizer);
  };
  return (
    <Box sx={{}}>
      <TextField
        select
        {...selectProps}
        id="optimizer-select"
        value={value?.classPath || ''}
        onChange={handleClassPathChange}
        label="Optimizer"
      >
        {optimizerOptions &&
          optimizerOptions.map((option) => (
            <MenuItem key={option.classPath} value={option.classPath}>
              {option.classPath}
            </MenuItem>
          ))}
      </TextField>

      {optionParams && (
        <Box>
          {Object.entries(optionParams).map(([key, paramSchema]) =>
            typeof paramSchema === 'string' ? null : (
              <TextField
                sx={selectProps?.sx || {}}
                type="number"
                key={key}
                label={paramSchema.label}
                // @ts-ignore
                inputProps={{ step: '0.001' }}
                // @ts-ignore
                value={value.params[key]}
                onChange={(event) => handleArgChange(event, key)}
              />
            )
          )}
        </Box>
      )}
    </Box>
  );
};

export default OptimizerForm;
