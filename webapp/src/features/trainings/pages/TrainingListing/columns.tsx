import {
  Box,
  Button,
  LinearProgress,
  Tooltip,
  Typography,
} from '@mui/material';
import { Experiment } from '@app/types/domain/experiments';
import AppLink from 'components/atoms/AppLink';
import Justify from 'components/atoms/Justify';
import { Column } from 'components/templates/Table';
import { dateRender } from 'components/atoms/Table/render';
import TrainingStatusChip from '@features/models/components/TrainingStatusChip';
import { Link } from 'react-router-dom';

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
    field: 'model',
    title: 'Model',
    name: 'Model',
    render: (row: Experiment) => (
      // !Ask backend for model info at experiments
      <Justify position="start">
        <Link to={`/models/${row.modelId}`}>
          <Typography>{row.model?.name}</Typography>
        </Link>
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
        <Typography variant="body2">(MSE)</Typography>
      </Box>
    ),
    render: (_row: Experiment, value: Experiment['trainMetrics']) => (
      <Justify position="start">
        {(() => {
          if (!value) return '-';
          else if ('train_loss' in value) {
            return value['train_loss'].toFixed(2);
          } else if ('trainLoss' in value) {
            return value['train_loss'].toFixed(2);
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
