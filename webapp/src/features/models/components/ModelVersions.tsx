import {
  TableActionsWrapper,
  tableActionsSx,
} from '@components/atoms/TableActions';
import StackTrace from '@components/organisms/StackTrace';
import { Check, Error, ReadMore } from '@mui/icons-material';
import { Button, CircularProgress, IconButton, Tooltip } from '@mui/material';
import { Box } from '@mui/system';
import { experimentsApi } from 'app/rtk/experiments';
import { Experiment } from 'app/types/domain/experiments';
import { ModelVersion } from 'app/types/domain/models';
import AppLink from 'components/atoms/AppLink';
import { dateRender } from 'components/atoms/Table/render';
import Table, { Column } from 'components/templates/Table';
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { sampleExperiment } from '../common';
import TrainingStatusChip from './TrainingStatusChip';
import Modal from '@components/templates/Modal';
interface ModelVersionItemProps {
  versions: ModelVersion[];
  modelId: number;
}

const ModelVersions = ({ modelId, versions }: ModelVersionItemProps) => {
  const { data: paginatedExperiments } = experimentsApi.useGetExperimentsQuery({
    modelId,
  });

  const navigate = useNavigate();

  const [selectedModelCheck, setSelectedModelCheck] = useState<ModelVersion>();

  const arrayDependancy = JSON.stringify(paginatedExperiments?.data);
  const experimentsByVersion = useMemo(() => {
    if (paginatedExperiments?.data)
      return paginatedExperiments?.data.reduce((acc, cur) => {
        if (cur.modelVersionId in acc) {
          acc[cur.modelVersionId].push(cur);
        } else {
          acc[cur.modelVersionId] = [cur];
        }
        return acc;
      }, {} as { [key: number]: Experiment[] });
  }, [arrayDependancy]);

  const handleModelCheckFix = (modelVersion: ModelVersion) => {
    navigate(`/models/${modelVersion.modelId}/${modelVersion.id}/fix`);
  };

  const columns: Column<
    ModelVersion,
    keyof ModelVersion,
    any,
    {
      experimentsByVersion: typeof experimentsByVersion;
    }
  >[] = [
    {
      field: 'name',
      title: 'Version',
      name: 'Version',
      render: (row: ModelVersion) => (
        <AppLink to={`/models/${row.modelId}/${row.id}`}>{row.name}</AppLink>
      ),
    },
    {
      title: 'Status',
      field: 'id',
      name: 'Training Status',
      render: (model: ModelVersion, value, { experimentsByVersion }) => {
        return !experimentsByVersion ? (
          <CircularProgress
            size="small"
            style={{ width: '40px', height: '40px' }}
          />
        ) : (
          <>
            <TrainingStatusChip trainings={experimentsByVersion[model.id]} />
          </>
        );
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
      valueGetter: (
        model,
        { experimentsByVersion }
      ): Experiment['stage'] | undefined => {
        if (!experimentsByVersion) return undefined;

        const modelVersionExperiments = experimentsByVersion[model.id];

        if (!modelVersionExperiments) return 'NOT RUNNING';

        const { successful, failed, running, notstarted } = sampleExperiment(
          modelVersionExperiments
        );

        if (successful) return 'SUCCESS';
        else if (running) return 'RUNNING';
        else if (failed) return 'ERROR';
        else if (notstarted) return 'NOT RUNNING';

        return undefined;
      },
    },
    {
      title: 'Created At',
      field: 'createdAt',
      name: 'Created At',
      render: dateRender((model: ModelVersion) => new Date(model.createdAt)),
    },
    {
      name: 'Actions',
      title: 'Actions',
      customSx: tableActionsSx,
      fixed: true,
      render: (model: ModelVersion) => {
        return (
          <TableActionsWrapper>
            <Button
              onClick={() => {
                setSelectedModelCheck(model);
              }}
              variant="text"
              color="primary"
              disabled={!model.checkStackTrace}
            >
              <ReadMore />
            </Button>
            {(() => {
              switch (model.checkStatus) {
                case 'FAILED':
                  return (
                    <Tooltip title="Navigate to fix model version">
                      <IconButton onClick={() => handleModelCheckFix(model)}>
                        <Error sx={{ color: '#ed6c02de' }} />
                      </IconButton>
                    </Tooltip>
                  );
                case 'OK':
                case null:
                  return <Check />;
                default:
                  return <CircularProgress size={25} />;
              }
            })()}
          </TableActionsWrapper>
        );
      },
    },
  ];

  return (
    <Box>
      <Modal
        open={!!selectedModelCheck}
        onClose={() => {
          setSelectedModelCheck(undefined);
        }}
        title="Failed model check"
      >
        <StackTrace
          stackTrace={selectedModelCheck?.checkStackTrace}
          message="Exception during model checking"
        />
      </Modal>
      <Table
        filterLinkOperatorOptions={['or']}
        rowKey={(row) => row.id}
        //TODO: Use TS to implement a way to the table component to accept column dependency types and pass it to the render function
        columns={columns as Column<any, any>[]}
        rows={versions}
        tableId="model-versions"
        usePreferences
        dependencies={{
          experimentsByVersion,
        }}
      />
    </Box>
  );
};

export default ModelVersions;
