export const BinnedDataHistogram = {
  data: {
    name: 'table',
  },
  title: 'Training Distribution of Molecular Weight',
  width: 400,
  height: 100,
  encoding: { x: { title: 'Molecular Weight' }, y: { title: 'Count' } },
  layer: [
    {
      mark: { type: 'bar' as const, color: '#384E77' },
      encoding: {
        x: { bin: { binned: true }, field: 'bin_start' },
        x2: { field: 'bin_end' },
        y: { field: 'count', type: 'quantitative' as const },
      },
    },
  ],
};
export const makeBinPlot = (
  overwrite: Partial<typeof BinnedDataHistogram> = {}
) => {
  return {
    ...BinnedDataHistogram,
    ...overwrite,
  };
};
