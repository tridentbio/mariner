import { Typography, capitalize } from '@mui/material';
import { useMemo } from 'react';
import { Vega, VisualizationSpec } from 'react-vega';
import styled from 'styled-components';
import { Options as TooltipOptions } from 'vega-tooltip';

const Container = styled.div`
  width: 45%;
  border: 1px solid rgba(0, 0, 0, 0.12);
  padding: 0.4rem;
  padding-top: 1rem;
  padding-bottom: 1rem;
  justify-content: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  border-radius: 4px;

  @media (max-width: 1371px) {
    width: 100%;
  }

  @media (min-width: 2400px) {
    width: 28%;
  }

  .MuiTypography-root {
    b {
      color: #384e77;
    }
  }
`;

const beautyMetrics = {
  val: 'Validation',
  train: 'Training',
};

const upperOrCapitalizeMetrics = (metric?: string) => {
  if (!metric) return '';
  if (['mse', 'mae', 'ev', 'mape'].includes(metric.toLowerCase()))
    return metric.toUpperCase();
  else return capitalize(metric);
};

const parseName = (name?: string): [string, string, string] => {
  if (!name || name.split('/').length < 3) return ['', '', ''];

  const [stage, metric, column] = name.split('/') as [
    keyof typeof beautyMetrics,
    string,
    string
  ];
  return [
    beautyMetrics[stage] || '',
    upperOrCapitalizeMetrics(metric).replace('2', '²'),
    column || '',
  ];
};

const tooltipConfigurator = (metric: string): TooltipOptions => ({
  formatTooltip: (val) => {
    const props = [
      { key: 'Epoch', value: val.Epoch },
      {
        key: metric,
        value: parseFloat(val[metric].replace('−', '-') || 0).toFixed(3),
      },
    ];
    if (!props.every((prop) => prop.value)) return '<div>Loading...</div>';
    return `
        <div >
            <style>
                .graphTooltip {
                    font-size: 20px;
                }
                b {
                    color: #384e77;
                }
            </style>
            
          ${props
            .map(
              (item) =>
                `<div class="graphTooltip"><b>${item.key}:</b> ${item.value}</div>`
            )
            .join('')}
        </div>
      `;
  },
});

type PlotProps = {
  epochs: number[];
  metricValues: number[];
  metricName: string;
};

export const MetricPlot = ({ epochs, metricValues, metricName }: PlotProps) => {
  const [stage, metric, column] = useMemo(
    () => parseName(metricName),
    [metricName]
  );

  const spec = useMemo<VisualizationSpec>(() => {
    return {
      $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
      width: 400,
      height: 230,
      data: {
        values: epochs.map((epoch, i) => ({ epoch, metric: metricValues[i] })),
      },
      encoding: {
        x: { field: 'epoch', type: 'quantitative' },
        y: { field: 'metric', type: 'quantitative' },
      },
      layer: [
        { mark: 'line' },
        {
          transform: [{ filter: { param: 'hover', empty: false } }],
          mark: { type: 'point', size: 50 },
        },
        {
          mark: 'rule',
          encoding: {
            opacity: {
              condition: { param: 'hover', empty: false, value: 0.3 },
              value: 0,
            },
            tooltip: [
              { field: 'epoch', type: 'quantitative', title: 'Epoch' },
              { field: 'metric', type: 'quantitative', title: metric },
            ],
          },
          params: [
            {
              name: 'hover',
              select: {
                type: 'point',
                fields: ['epoch', 'metric'],
                on: 'mouseover',
                clear: 'mouseout',
                nearest: true,
              },
            },
          ],
        },
        {
          transform: [{ filter: { param: 'hover', empty: false } }],
          mark: {
            type: 'tick',
            color: 'gray',
            strokeWidth: 2,
            orient: 'vertical',
          },
          encoding: {
            x: { field: 'epoch', type: 'quantitative', scale: { zero: false } },
            y: { field: 'metric', type: 'quantitative' },
            size: { value: 500 },
          },
        },
      ],

      config: {
        mark: { color: '#384e77' },
      },
    };
  }, [epochs, metricValues]);

  const tooltipConf = useMemo(() => tooltipConfigurator(metric), [metric]);

  return (
    <Container key={metricName}>
      <Typography
        variant="body1"
        dangerouslySetInnerHTML={{
          __html: `Metric: <b>${metric}</b>;  Target Column: <b>${column}</b>.`,
        }}
      />
      <br />
      <Vega spec={spec} tooltip={tooltipConf} />
    </Container>
  );
};
