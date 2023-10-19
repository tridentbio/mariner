import { useGetExperimentsMetricsQuery } from '@app/rtk/generated/experiments';
import { Experiment } from '@app/types/domain/experiments';
import Justify from '@components/atoms/Justify';
import { Column } from '@components/templates/Table';
import { Box, Typography } from '@mui/material';
import { useMemo } from 'react';

type ColumnT = Column<Experiment, keyof Experiment> & {
  isMetric?: boolean;
};

export default function useMetricsCols(experiments: Experiment[]) {
  const { data: metricsOptions } = useGetExperimentsMetricsQuery();
  const experimentsMergedMetrics = useMemo(() => {
    let merged: { [key: string]: any } = {};
    for (const experiment of experiments) {
      for (const metricProp of [
        'trainMetrics',
        'valMetrics',
        'testMetrics',
      ] as const) {
        if (!experiment[metricProp]) continue;
        for (const [key, value] of Object.entries(
          experiment[metricProp] || {}
        )) {
          if (experiment[metricProp] && experiment[metricProp]![key])
            merged[key] = value;
        }
      }
    }
    return merged;
  }, [experiments]);
  const makeMetricTitle = (metricKey: string) => {
    const [stage, metric, column] = metricKey.split('/');
    const stageTitle = stage === 'val' ? 'Validation' : 'Training';
    const metricTitle =
      metric === 'loss'
        ? 'Loss'
        : metricsOptions?.find((metric) => metric.key === metricKey)?.label ||
          metric;
    return `${stageTitle} ${metricTitle} (${column})`;
  };
  const makeMetricColumns = (
    metrics: typeof experimentsMergedMetrics
  ): ColumnT[] => {
    return Object.keys(metrics).map((key) => {
      const metricProp: 'valMetrics' | 'trainMetrics' | 'testMetrics' =
        key.includes('train')
          ? 'trainMetrics'
          : key.includes('val')
          ? 'valMetrics'
          : 'testMetrics';
      return {
        name: makeMetricTitle(key),
        isMetric: true,
        title: (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',

              alignItems: 'center',
            }}
          >
            <Typography>{makeMetricTitle(key)}</Typography>
          </Box>
        ),
        render: (row: Experiment) => {
          let value: number | string | undefined =
            row && row[metricProp] && row[metricProp]![key];
          if (typeof value === 'number') {
            value = value.toFixed(2);
          }
          return <Justify position="end">{value || null}</Justify>;
        },
        skeletonProps: {
          variant: 'text',
          width: 30,
        },
        customSx: {
          textAlign: 'center',
        },
      } as ColumnT;
    });
  };
  const columns = useMemo(
    () => makeMetricColumns(experimentsMergedMetrics),
    [experimentsMergedMetrics]
  );
  return columns;
}
