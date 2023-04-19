import * as modelsApi from 'app/types/domain/models';
import { useMemo } from 'react';
import { VisualizationSpec } from 'react-vega';
import { theme } from 'theme';

const colorScheme = {
  green: theme.palette.secondary.main,
  blue: theme.palette.primary.main,
};

export const modelOutputToVegaSpec = (
  outputs: modelsApi.ModelOutputValue
): VisualizationSpec => {
  const isMultiTargetColumns = useMemo(() => outputs.length > 1, [outputs]);
  const maxIndex = useMemo(
    () =>
      isMultiTargetColumns
        ? outputs.indexOf(Math.max(...(outputs as number[])))
        : 0,
    [outputs]
  );

  const encoding = useMemo(
    () =>
      isMultiTargetColumns
        ? {
            color: {
              field: 'Prediction',
              scale: {
                domain: Object.keys(outputs),
                range: outputs.map((_, index) =>
                  index === maxIndex
                    ? colorScheme['green']
                    : colorScheme['blue']
                ),
              },
            },
          }
        : {
            color: {
              field: 'Prediction',
              scale: {
                domain: Object.keys(outputs),
                range: [
                  outputs[0] > 0.5 ? colorScheme['green'] : colorScheme['blue'],
                ],
              },
            },
          },
    [outputs]
  );

  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    description:
      'Bar chart with text labels. Set domain to make the frame cover the labels.',
    data: {
      values: Object.entries(outputs).map(([key, value]) => ({
        Prediction: key,
        Probability: (value as number).toFixed(3),
      })),
    },
    encoding: {
      y: { field: 'Prediction', type: 'nominal' },
      x: {
        field: 'Probability',
        type: 'quantitative',
        scale: { domain: [0, 1] },
      },
    },
    layer: [
      {
        mark: 'bar',
        encoding,
      },
      {
        mark: {
          type: 'text',
          align: 'left',
          baseline: 'middle',
          dx: 3,
        },
        encoding: {
          text: { field: 'Probability', type: 'quantitative' },
        },
      },
    ],
  };
};
