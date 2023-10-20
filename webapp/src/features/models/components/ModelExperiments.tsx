import { ReadMore, Cancel } from '@mui/icons-material';
import Table, { Column } from 'components/templates/Table';
import { Model } from 'app/types/domain/models';
import { useEffect, useMemo, useRef, useState } from 'react';
import { dateRender } from 'components/atoms/Table/render';
import {
  Button,
  CircularProgress,
  IconButton,
  LinearProgress,
  Tooltip,
  Typography,
} from '@mui/material';
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
import { useGetExperimentsMetricsQuery } from '@app/rtk/generated/experiments';
import useMetricsCols from '@hooks/useMetricsCols';

interface ModelExperimentsProps {
  model: Model;
}

type QueryParams = Omit<FetchExperimentsQuery, 'modelId'>;

type ColumnT = Column<Experiment, keyof Experiment> & {
  isMetric?: boolean;
};

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

  const [selectedExperimentId, setSelectedExperimentId] = useState<
    number | undefined
  >();

  const [trainingCanceling, setTrainingCanceling] = useState(false);
  const [confirmExperimentCancelling, setConfirmExperimentCancelling] =
    useState<number | undefined>();
  const [cancelTraining] = experimentsApi.useCancelTrainingMutation();

  const handleCancelTraining = (experimentId: number) => {
    setTrainingCanceling(true);
    cancelTraining(experimentId).finally(() => {
      setTrainingCanceling(false);
    });
  };

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

  const metricsCols: ColumnT[] = useMetricsCols(storeExperiments);
  const actionsColumn = {
    name: 'Actions',
    title: 'Actions',
    customSx: tableActionsSx,
    fixed: true,
    render: (row: Experiment, value, { trainingCanceling }) => {
      return (
        <TableActionsWrapper>
          <Button
            onClick={() => {
              setSelectedExperimentId(row.id);
            }}
            variant="text"
            color="primary"
            disabled={!row.stackTrace}
          >
            <ReadMore />
          </Button>
          {row.stage === 'RUNNING' && (
            <Tooltip title="Cancel training" placement="top">
              <IconButton
                onClick={() => {
                  setConfirmExperimentCancelling(row.id);
                }}
                disabled={trainingCanceling}
              >
                <Cancel />
              </IconButton>
            </Tooltip>
          )}
        </TableActionsWrapper>
      );
    },
  } as ColumnT;

  const columns: ColumnT[] = [
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
      field: 'stage' as const,
      title: 'Stage',
      name: 'Stage',
      render: (row: Experiment) => (
        <Justify position="center">
          {row.stage === 'RUNNING' ? (
            row.progress !== 0 && !row.progress ? (
              <Tooltip title="Preparing to start">
                <CircularProgress size={30} />
              </Tooltip>
            ) : (
              <Tooltip title={`${((row.progress || 0) * 100).toFixed(2)}%`}>
                <LinearProgress
                  sx={{ minWidth: 100 }}
                  variant="determinate"
                  value={(row.progress || 0) * 100}
                />
              </Tooltip>
            )
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
          // @ts-ignore
          optionKey: (option) => option.key,
          // @ts-ignore
          getLabel: (option) => option.label,
        },
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
  ];
  const cols = columns.concat(metricsCols).concat([actionsColumn]);
  const columnsTreeView: TreeNode[] = [
    {
      id: 'attributes',
      name: 'Attributes',
      children: cols
        .filter((col) => !col.isMetric! && !col.fixed)
        .map((col) => ({
          id: col.name,
          name: col.name,
          parent: 'attributes',
        })),
    },
    {
      id: 'metrics',
      name: 'Metrics',
      children: cols
        .filter((col) => col.isMetric!)
        .map((col) => ({
          id: col.name,
          name: col.name,
          parent: 'metrics',
        })),
    },
  ];

  const detailedExperiment = useMemo(
    () =>
      selectedExperimentId === undefined
        ? undefined
        : experiments.find(
            (exp: Experiment) => exp?.id === selectedExperimentId
          ),
    [selectedExperimentId]
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
        open={selectedExperimentId !== undefined}
        onClose={() => {
          setSelectedExperimentId(undefined);
        }}
        title="Failed experiment error"
      >
        <StackTrace
          stackTrace={detailedExperiment?.stackTrace}
          message="Exception during model training"
        />
      </Modal>
      <ConfirmationDialog
        title="Confirm training cancellation"
        text={'Are you sure to cancel this training? '}
        alertText="You will not be able to recover this training once it is cancelled."
        confirmText="Yes, cancel it"
        cancelText="No, dismiss"
        onResult={(result) => {
          if (result == 'confirmed') {
            if (typeof confirmExperimentCancelling !== 'number') {
              // eslint-disable-next-line no-console
              console.error(
                '[ModelExperiments.tsx] Unexpected state: confirmExperimentCancelling is not a number'
              );
            } else {
              handleCancelTraining(confirmExperimentCancelling);
            }
          }
          setConfirmExperimentCancelling(undefined);
        }}
        open={confirmExperimentCancelling !== undefined}
      />
      <Table<any>
        onStateChange={handleTableStateChange}
        rows={storeExperiments}
        columns={cols}
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
