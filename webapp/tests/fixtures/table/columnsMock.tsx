import { Experiment } from '@app/types/domain/experiments';
import Justify from '@components/atoms/Justify';
import { dateRender } from '@components/atoms/Table/render';
import { TableActionsWrapper, tableActionsSx } from '@components/atoms/TableActions';
import { Column } from '@components/templates/Table';
import TrainingStatusChip from '@features/models/components/TrainingStatusChip';
import ReadMoreIcon from '@mui/icons-material/ReadMore';
import { Box, Button, LinearProgress, Tooltip, Typography } from '@mui/material';

export const columns = [
  {
    title: 'Experiment Name',
    name: 'Experiment Name',
    skeletonProps: {
      variant: 'text',
      width: 60,
    },
    field: 'experimentName', // @TODO: just a way of respecting the innacurate table interface
    render: (_: Experiment, value: string) => (
      <Justify position="start">{value}</Justify>
    ),
    filterSchema: {
      byIncludes: true
    }
  },
  {
    field: 'stage',
    title: 'Stage',
    name: 'Stage',
    render: (row: Experiment) => (
      <Justify position="center">
        {row.stage === 'RUNNING' ? (
          <Tooltip title={`${((row.progress || 0) * 100).toFixed(2)}%`}>
            <LinearProgress
              sx={{ minWidth: 100 }}
              variant="determinate"
              value={(row.progress || 0) * 100}
            />
          </Tooltip>
        ) : (
          <TrainingStatusChip trainings={[row]} />
        )}
      </Justify>
    ),
    skeletonProps: {
      variant: 'text',
      width: 30,
    },
    filterSchema: {
      byContains: {
        options: [
          { label: 'Trained', key: 'SUCCESS' },
          { label: 'Training', key: 'RUNNING' },
          { label: 'Failed', key: 'ERROR' },
          { label: 'Not started', key: 'NOT RUNNING' },
        ],
        optionKey: (option) => option.key,
        getLabel: (option) => option.label,
      },
    },
    customSx: {
      textAlign: 'center',
    },
  },
  {
    field: 'trainMetrics' as const,
    name: 'Train Loss',
    title: (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        <Typography>Train Loss</Typography>
      </Box>
    ),
    render: (_row: Experiment, value: Experiment['trainMetrics']) => (
      <Justify position="end">
        {(() => {
          if (!value || !_row) return '-';

          const column = _row.modelVersion.config.dataset.targetColumns[0];
          if (`train/loss/${column.name}` in value) {
            return value[`train/loss/${column.name}`].toFixed(2);
          }
        })()}
      </Justify>
    ),
    customSx: {
      textAlign: 'center',
    },
    skeletonProps: {
      variant: 'text',
      width: 30,
    },
  },
  {
    field: 'trainMetrics' as const,
    name: 'Validation Loss',
    title: (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',

          alignItems: 'center',
        }}
      >
        <Typography>Validation Loss</Typography>
      </Box>
    ),
    render: (_row: Experiment, value: Experiment['trainMetrics']) => (
      <Justify position="end">
        {(() => {
          if (!value || !_row) return '-';

          const column = _row.modelVersion.config.dataset.targetColumns[0];
          if (`val/loss/${column.name}` in value) {
            return value[`val/loss/${column.name}`].toFixed(2);
          }
        })()}
      </Justify>
    ),
    skeletonProps: {
      variant: 'text',
      width: 30,
    },
    customSx: {
      textAlign: 'center',
    },
  },
  {
    name: 'Learning Rate',
    field: 'hyperparams' as const,
    title: 'LR',
    render: (row: Experiment) => (
      <Justify position="end">{row.hyperparams?.learning_rate}</Justify>
    ),
    customSx: {
      textAlign: 'center',
    },
  },
  {
    name: 'Epochs',
    field: 'epochs' as const,
    title: 'Epochs',
    render: (row: Experiment) => (
      <Justify position="end">{row.epochs}</Justify>
    ),
    customSx: {
      textAlign: 'center',
    },
    sortable: true
  },
  {
    name: 'Created At',
    field: 'createdAt' as const,
    title: 'Created At',
    render: (exp: Experiment) => (
      <Justify position="start">
        {dateRender((exp: Experiment) => exp.createdAt)(exp)}
      </Justify>
    ),
    sortable: true,
    customSx: {
      textAlign: 'center',
    },
  },
  {
    name: 'Actions',
    title: 'Actions',
    customSx: tableActionsSx,
    fixed: true,
    render: (row: Experiment) => (
      <TableActionsWrapper>
        <Button
          onClick={() => {}}
          variant="text"
          color="primary"
          disabled={!row.stackTrace}
        >
          <ReadMoreIcon />
        </Button>
      </TableActionsWrapper>
    ),
  },
] as Column<Experiment, keyof Experiment>[]