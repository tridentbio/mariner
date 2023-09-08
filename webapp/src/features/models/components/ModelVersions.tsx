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
    },
    {
      title: 'Created At',
      field: 'createdAt',
      name: 'Created At',
      render: dateRender((model: ModelVersion) => new Date(model.createdAt)),
    },
  ];

  const makeFilterFunctionFromFilters =
    (filterModel: FilterModel) => (experiment: ModelVersion) => {
      type FilterRoutine = {
        [colField in Column<any, any>['field']]: (
          experiment: ModelVersion,
          op: OperatorValue,
          value: any
        ) => boolean;
      };

      const filterRoutines: FilterRoutine = {
        id: (experiment, op, value) => {
          if (op === 'ct') {
            if (!experimentsByVersion) return false;

            const experiments = experimentsByVersion[experiment.id];

            if (!experiments) return false;

            const { successful, failed, running, notstarted } =
              sampleExperiment(experiments);

            const result = value.some(
              (stage: Experiment['stage']) =>
                (stage === 'RUNNING' && !!running) ||
                (stage === 'NOT RUNNING' && !!notstarted) ||
                (stage === 'SUCCESS' && !!successful) ||
                (stage === 'ERROR' && !!failed)
            );

            return result;
          }

          return true;
        },
      };

      const result = filterModel.items.reduce((acc, item) => {
        if (filterRoutines[item.columnName])
          acc = filterRoutines[item.columnName](
            experiment,
            item.operatorValue,
            item.value
          );

        return acc;
      }, true);

      return result;
    };
  const handleFilterChange = (filterModel: FilterModel) => {
    setFilteredVersions(
      versions.filter(makeFilterFunctionFromFilters(filterModel))
    );
  };

  return (
    <Box>
      <Table
        onStateChange={(state) => handleFilterChange(state.filterModel)}
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
