import { CircularProgress } from '@mui/material';
import { Box } from '@mui/system';
import { experimentsApi } from 'app/rtk/experiments';
import { Experiment } from 'app/types/domain/experiments';
import { ModelVersion } from 'app/types/domain/models';
import AppLink from 'components/atoms/AppLink';
import Table, { Column } from 'components/templates/Table';
import { dateRender } from 'components/atoms/Table/render';
import { FilterModel, OperatorValue } from 'components/templates/Table/types';
import { useMemo, useState } from 'react';
import { sampleExperiment } from '../common';
import TrainingStatusChip from './TrainingStatusChip';
interface ModelVersionItemProps {
  versions: ModelVersion[];
  modelId: number;
}

const ModelVersions = ({ modelId, versions }: ModelVersionItemProps) => {
  const { data: paginatedExperiments } = experimentsApi.useGetExperimentsQuery({
    modelId,
  });
  const [filteredVersions, setFilteredVersions] =
    useState<ModelVersion[]>(versions);

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
  ];

  return (
    <Box>
      <Table
        filterLinkOperatorOptions={['or']}
        rowKey={(row) => row.id}
        //TODO: Use TS to implement a way to the table component to accept column dependency types and pass it to the render function
        columns={columns as Column<any, any>[]}
        rows={filteredVersions}
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
