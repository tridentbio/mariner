import {ReadMore, Cancel} from '@mui/icons-material';
import Table, { Column } from 'components/templates/Table';
import { Model } from 'app/types/domain/models';
import { useEffect, useMemo, useRef, useState } from 'react';
import { dateRender } from 'components/atoms/Table/render';
import { Button, IconButton, LinearProgress, Tooltip, Typography } from '@mui/material';
import Modal from 'components/templates/Modal';
import StackTrace from 'components/organisms/StackTrace';
import {
  Experiment,
  FetchExperimentsQuery,
} from 'app/types/domain/experiments';
import { Box } from '@mui/system';
import TrainingStatusChip from './TrainingStatusChip';
import { State } from 'components/templates/Table/types';
import { experimentsApi } from 'app/rtk/experiments';
import { useAppSelector } from 'app/hooks';
import { useDispatch } from 'react-redux';
import { updateExperiments } from '../modelSlice';
import Justify from 'components/atoms/Justify';
import {
  TableActionsWrapper,
  tableActionsSx,
} from '@components/atoms/TableActions';
import ConfirmationDialog from '@components/templates/ConfirmationDialog';

interface ModelExperimentsProps {
  model: Model;
}

type QueryParams = Omit<FetchExperimentsQuery, 'modelId'>;

const queryParamsInitialState = {
  perPage: 10,
  page: 0,
  orderBy: '-createdAt',
};

const ModelExperiments = ({ model }: ModelExperimentsProps) => {
  const [queryParams, setQueryParams] = useState<QueryParams>(
    queryParamsInitialState
  );

  const storeExperiments = useAppSelector((state) => state.models.experiments);
  const { data: paginatedExperiments } = experimentsApi.useGetExperimentsQuery({
    modelId: model.id,
    ...queryParams,
  });
  const { total } = useMemo(() => {
    return {
      experiments: paginatedExperiments?.data || [],
      total: paginatedExperiments?.total || 0,
    };
  }, [paginatedExperiments]);
  const experiments = useAppSelector((state) => state.models.experiments);

  const selectedExperimentId = useRef<number | undefined>();

  const showExperimentDetails = useRef<boolean>();

  const [trainingCanceling, setTrainingCanceling] = useState(false);
  const [showCancelConfirmation, setShowCancelConfirmation] = useState(false);
  const [cancelTraining] = experimentsApi.useCancelTrainingMutation()

  const handleCancelTraining = () => {
    setTrainingCanceling(true);

    cancelTraining(selectedExperimentId.current as number)
      .finally(() => {
        setTrainingCanceling(false);
      })
  }

  const handleTableStateChange = (state: State) => {
    const newQueryParams: QueryParams = {};

    if (state.paginationModel) {
      const { page, rowsPerPage: perPage } = state.paginationModel;

      if (page !== queryParams.page || perPage !== queryParams.perPage) {
        newQueryParams.page = page;
        newQueryParams.perPage = perPage;

        setQueryParams((prev) => ({ ...prev, ...newQueryParams }));
      }
    }
  };

  const columns: Column<Experiment, keyof Experiment>[] = [
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
      render: (row: Experiment, value, { trainingCanceling }) => {
        return (
          <TableActionsWrapper>
            <Button
              onClick={() => selectedExperimentId.current = row.id}
              variant="text"
              color="primary"
              disabled={!row.stackTrace}
            >
              <ReadMore />
            </Button>
            <Tooltip title="Cancel training" placement="top">
              <IconButton
                onClick={() => {
                  selectedExperimentId.current = row.id
                  setShowCancelConfirmation(true)
                }}
                disabled={trainingCanceling}
              >
                <Cancel />
              </IconButton>
            </Tooltip>
          </TableActionsWrapper>
        )
      },
    },
  ];

  const columnsTreeView: TreeNode[] = [
    {
      id: 'attributes',
      name: 'Attributes',
      children: columns
        .filter((col) => col.field != 'trainMetrics' && !col.fixed)
        .map((col) => ({
          id: col.name,
          name: col.name,
          parent: 'attributes',
        })),
    },
    {
      id: 'metrics',
      name: 'Metrics',
      children: columns
        .filter((col) => col.field === 'trainMetrics' && !col.fixed)
        .map((col) => ({
          id: col.name,
          name: col.name,
          parent: 'metrics',
        })),
    },
  ];

  const detailedExperiment = useMemo(
    () =>
      experiments.find((exp: Experiment) => exp?.id === selectedExperimentId.current),
    [selectedExperimentId.current]
  );
  const dispatch = useDispatch();
  useEffect(() => {
    if (paginatedExperiments?.data) {
      dispatch(updateExperiments(paginatedExperiments?.data));
    }
  }, [paginatedExperiments]);
  return (
    <div style={{ marginTop: 15 }}>
      <Modal
        open={showExperimentDetails.current !== undefined}
        onClose={() => showExperimentDetails.current = undefined}
        title="Failed experiment error"
      >
        <StackTrace
          stackTrace={detailedExperiment?.stackTrace}
          message="Exception during model training"
        />
      </Modal>
      <ConfirmationDialog
        title="Confirm training cancelation"
        text={'Are you sure to cancel this training? '}
        alertText="Be aware that won't be possible to recover it."
        onResult={result => {
          if(result == 'confirmed') handleCancelTraining()
        }}
        open={showCancelConfirmation}
        setOpen={setShowCancelConfirmation}
      />
      <Table<Experiment>
        onStateChange={handleTableStateChange}
        rows={storeExperiments}
        columns={columns}
        rowKey={(row) => row.id}
        rowAlign="center"
        pagination={{
          total,
          page: queryParams.page || 0,
          rowsPerPage: queryParams.perPage || 10,
        }}
        tableId="model-experiments"
        usePreferences
        columnTree={columnsTreeView}
        defaultSelectedNodes={['attributes', 'metrics']}
        dependencies={{
          trainingCanceling,
        }}
      />
    </div>
  );
};

export default ModelExperiments;
