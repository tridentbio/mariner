import * as modelsApi from 'app/types/domain/models';
import { useMemo } from 'react';
import { VisualizationSpec } from 'react-vega';
import { theme } from 'theme';

const colorScheme = {
  green: theme.palette.secondary.main,
  blue: theme.palette.primary.main,
};

export const modelOutputToVegaSpec = (
  outputs: modelsApi.ModelOutputValue,
  classIndices?: (number | string)[],
): VisualizationSpec => {
  let flatOutputs = outputs.flat();
  if (flatOutputs.length === 1 && typeof flatOutputs[0] === 'number') {
    flatOutputs = [1 - flatOutputs[0], flatOutputs[0]];
  }
  const isMultiTargetColumns = useMemo(
    () => flatOutputs.length > 1,
    [flatOutputs],
  );
  const maxIndex = useMemo(
    () =>
      isMultiTargetColumns
        ? flatOutputs.indexOf(Math.max(...(flatOutputs as number[])))
        : 0,
    [flatOutputs],
  );

  const encoding = useMemo(
    () =>
      isMultiTargetColumns
        ? {
            color: {
              field: 'Prediction',
              scale: {
                domain: classIndices || Object.keys(flatOutputs),
                range: flatOutputs.map((_, index) =>
                  index === maxIndex
                    ? colorScheme['green']
                    : colorScheme['blue'],
                ),
              },
            },
          }
        : {
            color: {
              field: 'Prediction',
              scale: {
                domain: Object.keys(flatOutputs),
                range: [
                  flatOutputs[0] > 0.5
                    ? colorScheme['green']
                    : colorScheme['blue'],
                ],
              },
            },
          },
    [flatOutputs, classIndices],
  );

  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    description:
      'Bar chart with text labels. Set domain to make the frame cover the labels.',
    data: {
      values: Object.entries(flatOutputs).map(([key, value]) => ({
        Prediction: classIndices ? classIndices[key] : key,
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
