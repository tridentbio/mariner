import { FormControlLabel, Radio, RadioGroup } from '@mui/material';
import {
  DataSummary as APIDataSummary,
  Plot,
  PlotSmiles,
} from 'app/types/domain/datasets';
import { TrueFalseChart } from 'features/datasets/components/charts/trueFalseChart';
import { extractVal } from 'features/models/common';
import { useState } from 'react';
import { VegaLite } from 'react-vega';
import { keys } from 'utils';
import { datasetsGraphTitlesMapper } from './utils';

export interface DataSummaryProps {
  columnsData: {
    train: APIDataSummary;
    val: APIDataSummary;
    test: APIDataSummary;
    full: APIDataSummary;
  };
  titlePrefix?: string;
  inference?: {
    columnName: string;
    value: number;
  }[];
}

type PlotTitles = {
  [key in keyof Plot]: string;
};
const Plots = (props: {
  col: string;
  plots: Plot;
  titles: PlotTitles;
  inferenceValue?: number;
}) => {
  const hasLabel = Array.isArray(props.plots.hist?.values)
    ? props.plots.hist.values[0].hasOwnProperty('label')
    : false;
  return (
    <div>
      {keys(props.plots).map((data, index) => {
        if (data === 'hist') {
          const values = props.plots.hist.values.map((value: any, index) => {
            if (hasLabel)
              return {
                label: value.label,
                count: value.count,
              };

            if (index == props.plots.hist.values.length - 1)
              return {
                max: value.max,
                mean: value.mean,
                median: value.median,
                min: value.min,
              };

            return {
              bin_start: value.bin_start,
              bin_end: value.bin_end,
              count: value.count,
            };
          });
          return (
            <VegaLite
              key={data + index}
              actions={false}
              spec={{
                $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
                data: {
                  values: values,
                },
                title: props.titles.hist,
                width: 400,
                height: 100,
                encoding: { x: { title: props.col }, y: { title: 'Count' } },
                layer: hasLabel
                  ? [
                      {
                        mark: { type: 'bar', color: '#384E77' },
                        encoding: {
                          x: { field: 'label', type: 'nominal' },
                          y: { field: 'count', type: 'quantitative' },
                        },
                      },
                    ]
                  : [
                      {
                        mark: { type: 'bar', color: '#384E77' },
                        encoding: {
                          x: { bin: { binned: true }, field: 'bin_start' },
                          x2: { field: 'bin_end' },
                          y: { field: 'count', type: 'quantitative' },
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
                            field: 'bin_end',
                            aggregate: 'max',
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
                            field: 'bin_start',
                            aggregate: 'min',
                          },
                        },
                      },
                      ...(props.inferenceValue != undefined
                        ? [
                            {
                              mark: {
                                type: 'rule' as const,
                                strokeWidth: 5,
                                color: '#E6F9AF',
                              },
                              encoding: {
                                x: {
                                  datum: props.inferenceValue,
                                },
                              },
                            },
                          ]
                        : []),
                    ],
              }}
            />
          );
        } else {
          return null;
        }
      })}
    </div>
  );
};

const DataSummary = (props: DataSummaryProps) => {
  const [split, setSplit] = useState<keyof typeof props.columnsData>('full');
  const isFlatPlot = (obj: Plot | { [key: string]: Plot }): obj is Plot =>
    'hist' in obj;
  const title = 'Distribution of';

  return (
    <div>
      <RadioGroup
        aria-labelledby="demo-radio-buttons-group-label"
        defaultValue={split}
        value={split}
        onChange={(v) =>
          setSplit(v.target.value as keyof typeof props.columnsData)
        }
        name="radio-buttons-group"
        sx={{ display: 'flex', flexDirection: 'row' }}
      >
        <FormControlLabel
          value="train"
          control={<Radio />}
          label="Training data"
        />

        <FormControlLabel
          value="val"
          control={<Radio />}
          label="Validation data"
        />
        <FormControlLabel value="test" control={<Radio />} label="Test data" />
        <FormControlLabel value="full" control={<Radio />} label="All data" />
      </RadioGroup>
      <div style={{ display: 'flex', flexWrap: 'wrap' }}>
        {keys(props.columnsData[split]).map((col, index) => {
          const thing: Plot = { ...props.columnsData[split][col] };
          const inferenceObject = (props.inference || []).find(
            ({ columnName }) => columnName === col
          );
          const inferenceValue = inferenceObject
            ? extractVal(inferenceObject.value)
            : undefined;
          const titles = {
            hist: props.titlePrefix
              ? `${props.titlePrefix} ${title} ${
                  datasetsGraphTitlesMapper[
                    col as keyof typeof datasetsGraphTitlesMapper
                  ] || col
                }`
              : `${title} ${
                  datasetsGraphTitlesMapper[
                    col as keyof typeof datasetsGraphTitlesMapper
                  ] || col
                }`,
          };
          return isFlatPlot(thing) ? (
            <Plots
              key={String(col) + index}
              col={
                datasetsGraphTitlesMapper[
                  col as keyof typeof datasetsGraphTitlesMapper
                ] || (col as string)
              }
              titles={titles}
              plots={thing}
              inferenceValue={inferenceValue}
            />
          ) : (
            keys(thing).map((tKey, index) => {
              const inferenceObject = (props.inference || []).find(
                ({ columnName }) => columnName === tKey
              );
              const inferenceValue = inferenceObject
                ? extractVal(inferenceObject.value)
                : undefined;
              const xAxisName =
                datasetsGraphTitlesMapper[
                  tKey as keyof typeof datasetsGraphTitlesMapper
                ];

              if (tKey === 'has_chiral_centers') {
                return (
                  <TrueFalseChart
                    key={tKey + index}
                    col={xAxisName}
                    title={`${titles.hist}'s "${xAxisName}"`}
                    values={
                      (thing as PlotSmiles)[tKey].hist.values.slice(0, 2) as {
                        bin_start: number;
                        bin_end: number;
                        count: number;
                      }[]
                    }
                  />
                );
              }

              return (
                <Plots
                  col={xAxisName}
                  key={`${col}-${String(tKey)}` + index}
                  titles={{
                    ...titles,
                    hist: `${titles.hist}'s "${
                      datasetsGraphTitlesMapper[
                        tKey as keyof typeof datasetsGraphTitlesMapper
                      ]
                    }"`,
                  }}
                  plots={thing[tKey]}
                  inferenceValue={inferenceValue}
                />
              );
            })
          );
        })}
      </div>
    </div>
  );
};

export default DataSummary;
