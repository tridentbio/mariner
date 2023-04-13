import { VisualizationSpec } from 'react-vega';

export const CategoricalHistogram: VisualizationSpec = {
  $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
  data: { url: 'data/movies.json' },
  title: 'Training Distribution of <property name>',
  width: 400,
  height: 100,
  encoding: {
    x: {
      title: '<property name> (<units>)',
    },
    y: {
      title: 'Count',
    },
  },
  layer: [
    {
      mark: {
        type: 'bar',
        color: 'gray',
      },
      encoding: {
        x: {
          field: 'MPAA Rating',
        },
        y: { aggregate: 'count' },
      },
    },
    {
      mark: {
        type: 'bar',
        color: 'red',
      },
      transform: [{ filter: "datum['MPAA Rating'] == 'Not Rated'" }],
      encoding: {
        x: {
          field: 'MPAA Rating',
        },
        y: { aggregate: 'count' },
      },
    },

    {
      mark: {
        type: 'rule',
        strokeWidth: 3,
        strokeDash: [5, 3],
        color: 'red',
      },
      encoding: {
        x: {
          datum: 'Not Rated',
        },
      },
    },
  ],
};
