import {
  Box,
  Button,
  LinearProgress,
  Tooltip,
  Typography,
} from '@mui/material';
import { Experiment } from 'app/types/domain/experiments';
import AppLink from 'components/atoms/AppLink';
import Justify from 'components/atoms/Justify';
import { Column } from 'components/templates/Table';
import { dateRender } from 'components/atoms/Table/render';
import TrainingStatusChip from 'features/models/components/TrainingStatusChip';
import { Link } from 'react-router-dom';

const makeMetric = (
  title: string,
  field: 'trainMetrics' | 'valMetrics' | 'testMetrics'
): Column<Experiment, keyof Experiment> => ({
  field,
  name: 'Train Loss',
  title: (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <Typography>{title}</Typography>
    </Box>
  ),
  render: (_row: Experiment, value: Experiment[typeof field]) => {
    const targetColumns = _row.modelVersion.config.dataset.targetColumns;
    const targetColumn = targetColumns[0];
    const dataset = field.replace('Metrics', '');
    const lossKey = `${dataset}/loss/${targetColumn.name}`;
    let loss = '';
    if ('lossFn' in targetColumn && targetColumn.lossFn) {
      // @ts-ignore
      loss = targetColumn.lossFn.replace('torch.nn.', '').replace('Loss', '');
    }
    let text = '';
    if (value && lossKey in value) text = value[lossKey].toFixed(2);
    return (
      <Justify position="start">
        {text}
        {'\n'}({loss})
      </Justify>
    );
  },
  skeletonProps: {
    variant: 'text',
    width: 30,
  },
  customSx: {
    textAlign: 'center',
  },
});

export const trainingListingColumns: Column<Experiment, keyof Experiment>[] = [
  {
    title: 'Experiment Name',
    name: 'Experiment Name',
    skeletonProps: {
      variant: 'text',
      width: 60,
    },
    field: 'experimentName',
    render: (_: Experiment, value: string) => (
      <Justify position="start">{value}</Justify>
    ),
  },
  {
    title: 'Dataset',
    name: 'Dataset',
    render: (row: Experiment) => (
      // !Ask backend for datasetId at experiments
      // <Link to={`/datasets/${row.modelVersion?.config?.dataset.id}`}>
      <Justify position="start">
        <Typography>{row.modelVersion?.config?.dataset.name}</Typography>
      </Justify>
      // </Link>
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
    field: 'modelVersion',
    title: 'Model Version',
    name: 'Model version',
    render: (row: Experiment) => (
      <Justify position="start">
        <AppLink
          to={`/models/${row.modelVersion?.modelId}/${row.modelVersionId}`}
        >
          <Typography>{row.modelVersion?.name}</Typography>
        </AppLink>
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
    field: 'stage',
    title: 'Stage',
    name: 'Stage',
    render: (row: Experiment) => (
      <Justify position="center">
        <TrainingStatusChip trainings={[row]} />
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
  makeMetric('Train Loss', 'trainMetrics'),
  makeMetric('Validation Loss', 'valMetrics'),
  {
    name: 'Created At',
    field: 'createdAt' as const,
    title: 'Created At',
    render: (exp: Experiment) => (
      <Justify position="start">
        {dateRender((exp: Experiment) => exp.createdAt)(exp)}
      </Justify>
    ),
    customSx: {
      textAlign: 'center',
    },
    sortable: true,
  },
];
