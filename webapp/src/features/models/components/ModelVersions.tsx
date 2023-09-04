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
  const columns: Column<ModelVersion, keyof ModelVersion>[] = [
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
      render: (model: ModelVersion) =>
        !experimentsByVersion ? (
          <CircularProgress
            size="small"
            style={{ width: '40px', height: '40px' }}
          />
        ) : (
          <>
            <TrainingStatusChip trainings={experimentsByVersion[model.id]} />
          </>
        ),
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
      const obj = {
        'Training Status': (
          experiment: ModelVersion,
          op: OperatorValue,
          value: any
        ) => {
          if (op === 'ct') {
            if (!experimentsByVersion) return false;
            const experiments = experimentsByVersion[experiment.id];
            if (!experiments) return false;
            const { successful, failed, running, notstarted } =
              sampleExperiment(experiments);
            const result = value.some((stage: Experiment['stage']) => {
              (stage === 'RUNNING' && !!running) ||
                (stage === 'NOT RUNNING' && !!notstarted) ||
                (stage === 'SUCCESS' && !!successful) ||
                (stage === 'ERROR' && !!failed);
            });
            return result;
          }
          return true;
        },
      };
      return filterModel.items.reduce((acc, item) => {
        return (
          (acc && item.columnName !== 'id') ||
          (acc &&
            item.columnName === 'Training Status' &&
            obj[item.columnName](experiment, item.operatorValue, item.value))
        );
      }, true);
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
        columns={columns}
        rows={filteredVersions}
        tableId="model-versions"
        usePreferences
      />
    </Box>
  );
};

export default ModelVersions;
