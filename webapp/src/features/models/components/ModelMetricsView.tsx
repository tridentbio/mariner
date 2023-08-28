import { useGetExperimentsMetricsQuery } from '@app/rtk/generated/experiments';
import Table, { Column, TableProps } from '@components/templates/Table';
import { Box, Divider, SxProps } from '@mui/material';
import {
  Experiment,
  Model,
  useGetExperimentsMetricsForModelVersionQuery,
} from 'app/rtk/generated/models';
import Select from 'components/molecules/CenteredSelect';
import { MetricsAccordionPlot } from 'components/templates/MetricsAccordionPlot';
import { useState } from 'react';
import ModelVersionSelect from './ModelVersionSelect';

type ModelMetricsViewProps = {
  model: Model;
};

const containerSx: SxProps = {
  display: 'flex',
  flexDirection: 'row',
  gap: '2rem',
  justifyContent: 'space-between',
};

const FrameworkExperimentMetrics = ({
  experiment,
  model,
}: {
  experiment: Experiment;
  model: Model;
}) => {
  const experimentModelVersion = model.versions.find(
    (version) => version.id === experiment.modelVersionId
  );

  if (!experimentModelVersion) return null;

  switch (experimentModelVersion.config.framework) {
    case 'torch':
      return <MetricsAccordionPlot experiment={experiment} />;
    default:
      return null;
  }
};
interface MetricsTableProps {
  trainMetrics?: Record<string, string | number> | null;
  valMetrics?: Record<string, string | number> | null;
  testMetrics?: Record<string, string | number> | null;
}
type MetricEntry = {
  key: string;
  value: string | number;
  stage: 'val' | 'train' | 'test';
};
const getMetricStage = (key: string) => {
  const firstSlashIndex = key.indexOf('/');
  return key.substring(0, firstSlashIndex) as 'val' | 'train' | 'test';
};
const getMetric = (key: string) => {
  const firstSlashIndex = key.indexOf('/');
  const secondSlashIndex = key.indexOf('/', firstSlashIndex + 1);
  return key.substring(firstSlashIndex + 1, secondSlashIndex);
};
const sortRows = <K extends object>(
  rows: K[],
  sortModel: TableProps<K>['sortModel']
) => {
  return rows.sort((a, b) => {
    const order = 0;
    for (const sortItem of sortModel || []) {
      const { sort, field } = sortItem;
      // @ts-ignore
      const aValue = a[field];
      // @ts-ignore
      const bValue = b[field];
      if (aValue === bValue) continue;
      if (sort === 'asc') {
        if (aValue > bValue) return 1;
        else return -1;
      } else {
        if (aValue > bValue) return -1;
        else return 1;
      }
    }
    return order;
  });
};
const MetricsTable = ({
  trainMetrics,
  valMetrics,
  testMetrics,
}: MetricsTableProps) => {
  const { data: metrics } = useGetExperimentsMetricsQuery();
  const rows: MetricEntry[] = Object.entries(trainMetrics || {})
    .concat(Object.entries(valMetrics || {}))
    .concat(Object.entries(testMetrics || {}))
    .map(([key, value]) => ({ key, value, stage: getMetricStage(key) }));
  const columns: Column<MetricEntry, keyof MetricEntry>[] = [
    {
      title: 'Stage',
      name: 'stage',
      field: 'stage',
      render: (_, value) => {
        if (value === 'train') return 'Train';
        else if (value === 'val') return 'Validation';
        else if (value === 'test') return 'Test';
      },
      sortable: true,
    },
    {
      title: 'Metric Key',
      name: 'key',
      field: 'key',
      render: (_, metricKey) => {
        const metric = getMetric(metricKey);
        if (!metrics) return metric;
        else {
          const metricEntry = metrics.find((m) => m.key === metric);
          const label =
            metricEntry?.texLabel?.replace('^2', '²') ||
            metricEntry?.label ||
            metric;
          return label;
        }
      },
      sortable: true,
    },
    {
      title: 'Value',
      name: 'value',
      type: 'text',
      field: 'value',
      sortable: true,
    },
  ];
  const [sortModel, setSortModel] = useState<TableProps<any>['sortModel']>([]);
  return (
    <Table<MetricEntry>
      onStateChange={({ sortModel }) => {
        setSortModel(sortModel);
      }}
      rowKey={(row) => row.key}
      sortingMode="client"
      sortModel={sortModel}
      columns={columns}
      rows={sortRows(rows, sortModel)}
    />
  );
};
const ModelMetricsView = ({ model }: ModelMetricsViewProps) => {
  const [selectedModelVersionId, setSelectedModelVersionId] =
    useState<number>(-1);

  const [selectedExperiment, setSelectedExperiment] =
    useState<Experiment | null>(null);

  const { data: currentExperiments = [] } =
    useGetExperimentsMetricsForModelVersionQuery({
      modelVersionId: selectedModelVersionId,
    });

  const onModelVersionChange = (modelVersionId: number) => {
    setSelectedModelVersionId(modelVersionId);
    setSelectedExperiment(null);
  };

  return (
    <>
      <Box sx={containerSx}>
        <ModelVersionSelect
          sx={{ width: '100%', display: 'flex', alignItems: 'end' }}
          disableClearable
          model={model}
          value={model.versions.find(
            (version) => version.id === selectedModelVersionId
          )}
          onChange={(modelVersion) =>
            modelVersion && onModelVersionChange(modelVersion.id)
          }
        />

        <Select
          sx={{ width: '100%' }}
          title="Experiment"
          disabled={!currentExperiments.length}
          items={currentExperiments}
          keys={{ value: 'experimentName', children: 'experimentName' }}
          value={selectedExperiment?.experimentName || ''}
          data-testid="experiment-select"
          onChange={({ target }) => {
            setSelectedExperiment(
              currentExperiments.find(
                (experiment) => experiment.experimentName === target.value
              ) || null
            );
          }}
        />
      </Box>
      <Divider sx={{ my: '2rem' }} />
      {selectedExperiment && selectedExperiment.history && (
        <FrameworkExperimentMetrics
          model={model}
          experiment={selectedExperiment}
        />
      )}

      {selectedExperiment && (
        <MetricsTable
          trainMetrics={selectedExperiment.trainMetrics}
          valMetrics={selectedExperiment.valMetrics}
          testMetrics={selectedExperiment.testMetrics}
        />
      )}
    </>
  );
};

export default ModelMetricsView;
