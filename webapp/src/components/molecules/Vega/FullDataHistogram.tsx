import { VisualizationSpec } from 'react-vega';

// Full Data Histogram
export const histogramSpec: VisualizationSpec = {
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
        type: 'rule',
        color: 'black',
        strokeWidth: 3,
        strokeDash: [5, 3],
      },
      encoding: {
        x: {
          field: 'IMDB Rating',
          aggregate: 'min',
        },
      },
    },
    {
      mark: {
        type: 'rule',
        color: 'black',
        strokeWidth: 3,
        strokeDash: [5, 3],
      },
      encoding: {
        x: {
          field: 'IMDB Rating',
          aggregate: 'max',
        },
      },
    },
    {
      mark: {
        type: 'bar',
        color: '#384E77',
      },
      encoding: {
        x: {
          bin: true,
          field: 'IMDB Rating',
        },
        y: { aggregate: 'count' },
      },
    },
    {
      mark: {
        type: 'rule',
        strokeWidth: 5,
        color: '#E6F9AF',
      },
      encoding: {
        x: {
          datum: 4.3,
        },
      },
    },
  ],
};
