import { VegaLite } from 'react-vega';

export const TrueFalseChart = (props: {
  col: string;
  title: string;
  inferenceValue?: number;
  values: { bin_start: number; bin_end: number; count: number }[];
}) => {
  const values = props.values.map((value) => {
    if (value.bin_start >= -0.5 && value.bin_end <= 0.5) {
      return { category: 'false', amount: value.count };
    }
    if (value.bin_start >= 0.5 && value.bin_end <= 1.5) {
      return { category: 'true', amount: value.count };
    }
  });

  return (
    <VegaLite
      actions={false}
      spec={{
        $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
        data: {
          values: values,
        },
        title: props.title,
        width: 400,
        height: 100,
        encoding: { x: { title: props.col }, y: { title: 'Count' } },
        layer: [
          {
            mark: { type: 'bar', color: '#384E77' },
            encoding: {
              x: { field: 'category', axis: { labelAngle: 0 } },
              y: { field: 'amount', type: 'quantitative' },
            },
          },
        ],
      }}
    />
  );
};
