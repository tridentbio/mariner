import { store } from '@app/store';
import Justify from '@components/atoms/Justify';
import { TableActionsWrapper } from '@components/atoms/TableActions';
import { useAppSelector } from '@hooks';
import { Box, Button, ThemeProvider, Typography } from '@mui/material';
import { StoryFn, StoryObj } from '@storybook/react';
import { Provider } from 'react-redux';
import { theme } from 'theme';
import Table, { Column } from '.';
import { experimentsApi } from '@app/rtk/experiments';

export default {
  title: 'components/Table',
  component: Table,
  decorators: [
    (Story: StoryFn) => {
      return (
        <Provider store={store}>
          <ThemeProvider theme={theme}>
            <Story />
          </ThemeProvider>
        </Provider>
      );
    },
  ],
  args: {
    value: {
      columns: [
        {
          title: 'Experiment Name',
          name: 'Experiment Name',
          skeletonProps: {
            variant: 'text',
            width: 60,
          },
          field: 'experimentName', // @TODO: just a way of respecting the innacurate table interface
          render: (_: any, value: string) => (
            <Justify position="start">{value}</Justify>
          ),
        },
        {
          field: 'stage',
          title: 'Stage',
          name: 'Stage',
          render: (row: any) => <Justify position="center">teste</Justify>,
          skeletonProps: {
            variant: 'text',
            width: 30,
          },
          filterSchema: {
            byContains: {
              options: [
                { label: 'Trained', key: 'SUCCESS' },
                { label: 'Training', key: 'RUNNING' },
                { label: 'Failed', key: 'ERROR' },
                { label: 'Not started', key: 'NOT RUNNING' },
              ],
              optionKey: (option) => option.key,
              getLabel: (option) => option.label,
            },
          },
          customSx: {
            textAlign: 'center',
          },
        },
        {
          field: 'trainMetrics' as const,
          name: 'Train Loss',
          title: (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
              }}
            >
              <Typography>Train Loss</Typography>
            </Box>
          ),
          render: (_row: any, value: any['trainMetrics']) => (
            <Justify position="end">
              {(() => {
                if (!value || !_row) return '-';

                const column =
                  _row.modelVersion.config.dataset.targetColumns[0];
                if (`train/loss/${column.name}` in value) {
                  return value[`train/loss/${column.name}`].toFixed(2);
                }
              })()}
            </Justify>
          ),
          customSx: {
            textAlign: 'center',
          },
          skeletonProps: {
            variant: 'text',
            width: 30,
          },
        },
        {
          field: 'trainMetrics' as const,
          name: 'Validation Loss',
          title: (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',

                alignItems: 'center',
              }}
            >
              <Typography>Validation Loss</Typography>
            </Box>
          ),
          render: (_row: any, value: any['trainMetrics']) => (
            <Justify position="end">
              {(() => {
                if (!value || !_row) return '-';

                const column =
                  _row.modelVersion.config.dataset.targetColumns[0];
                if (`val/loss/${column.name}` in value) {
                  return value[`val/loss/${column.name}`].toFixed(2);
                }
              })()}
            </Justify>
          ),
          skeletonProps: {
            variant: 'text',
            width: 30,
          },
          customSx: {
            textAlign: 'center',
          },
        },
        {
          name: 'Learning Rate',
          field: 'hyperparams' as const,
          title: 'LR',
          render: (row: any) => (
            <Justify position="end">{row.hyperparams?.learning_rate}</Justify>
          ),
          customSx: {
            textAlign: 'center',
          },
        },
        {
          name: 'Epochs',
          field: 'epochs' as const,
          title: 'Epochs',
          render: (row: any) => <Justify position="end">{row.epochs}</Justify>,
          customSx: {
            textAlign: 'center',
          },
        },
        {
          name: 'Created At',
          field: 'createdAt' as const,
          title: 'Created At',
          render: (exp: any) => (
            <Justify position="start">{new Date().toString()}</Justify>
          ),
          sortable: true,
          customSx: {
            textAlign: 'center',
          },
        },
        {
          name: 'Actions',
          title: 'Actions',
          //   customSx: tableActionsSx,
          render: (row: any) => (
            <TableActionsWrapper>
              <Button
                // onClick={() => setExperimentDetailedId(row.id)}
                variant="text"
                color="primary"
                disabled={!row.stackTrace}
              >
                test
              </Button>
            </TableActionsWrapper>
          ),
        },
      ] as Column<any, any>[],
    },
  },
};

export const Simple: StoryObj = {
  render: ({ value }: { value?: { columns: Column<any, any>[] } }) => {
    const paginatedExperiments = [
      {
        experimentName: 'teste 3',
        modelVersionId: 1,
        modelVersion: {
          id: 1,
          modelId: 1,
          name: 'test model version',
          description: 'fwscttrs',
          mlflowVersion: '20',
          mlflowModelName:
            '1-SOME_MODEL_NAME-dfdada37-9a7f-4555-8670-5c697eedaf07',
          config: {
            name: 'test model version',
            framework: 'torch',
            spec: {
              layers: [
                {
                  type: 'fleet.model_builder.layers.Concat',
                  name: 'Concat-0',
                  constructorArgs: {
                    dim: 1,
                  },
                  forwardArgs: {
                    xs: ['$sepal_length', '$sepal_width'],
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-1',
                  constructorArgs: {
                    in_features: 2,
                    out_features: 16,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$Concat-0',
                  },
                },
                {
                  type: 'torch.nn.ReLU',
                  name: 'ReLU-2',
                  constructorArgs: {
                    inplace: false,
                  },
                  forwardArgs: {
                    input: '$Linear-1',
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-3',
                  constructorArgs: {
                    in_features: 16,
                    out_features: 16,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$ReLU-2',
                  },
                },
                {
                  type: 'torch.nn.ReLU',
                  name: 'ReLU-4',
                  constructorArgs: {
                    inplace: false,
                  },
                  forwardArgs: {
                    input: '$Linear-3',
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-5',
                  constructorArgs: {
                    in_features: 16,
                    out_features: 1,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$ReLU-4',
                  },
                },
              ],
            },
            dataset: {
              name: 'IRIS_DATASET_NAME',
              strategy: 'forwardArgs',
              targetColumns: [
                {
                  name: 'large_petal_length',
                  dataType: {
                    domainKind: 'categorical',
                    classes: {
                      '0': 0,
                      '1': 1,
                    },
                  },
                  outModule: 'Linear-5',
                  lossFn: 'torch.nn.BCEWithLogitsLoss',
                  columnType: 'binary',
                },
              ],
              featureColumns: [
                {
                  name: 'sepal_length',
                  dataType: {
                    domainKind: 'numeric',
                    unit: 'mole',
                  },
                },
                {
                  name: 'sepal_width',
                  dataType: {
                    domainKind: 'numeric',
                    unit: 'mole',
                  },
                },
              ],
              featurizers: [],
              transforms: [],
            },
          },
          createdAt: '2023-08-16T15:58:16.738343+00:00',
          updatedAt: '2023-08-16T15:58:16.738343',
        },
        createdAt: '2023-08-28T19:50:57.776251+00:00',
        updatedAt: '2023-08-28T19:51:14.400114+00:00',
        createdById: 1,
        id: 127,
        mlflowId: null,
        stage: 'ERROR',
        createdBy: {
          email: 'admin@mariner.trident.bio',
          isActive: true,
          isSuperuser: true,
          fullName: null,
          id: 1,
        },
        hyperparams: {
          learning_rate: 0.001,
          epochs: 299,
        },
        epochs: null,
        trainMetrics: null,
        valMetrics: null,
        testMetrics: null,
        history: null,
        stackTrace:
          '\u001b[36mray::TrainingActor.fit()\u001b[39m (pid=4250, ip=172.63.0.4, repr=<fleet.ray_actors.training_actors.TrainingActor object at 0xfffe1917eca0>)\n  File "/app/fleet/ray_actors/training_actors.py", line 48, in fit\n    return fit(**args)\nTypeError: fit() got an unexpected keyword argument \'epochs\'',
      },
      {
        experimentName: 'teste 3',
        modelVersionId: 1,
        modelVersion: {
          id: 1,
          modelId: 1,
          name: 'test model version',
          description: 'fwscttrs',
          mlflowVersion: '20',
          mlflowModelName:
            '1-SOME_MODEL_NAME-dfdada37-9a7f-4555-8670-5c697eedaf07',
          config: {
            name: 'test model version',
            framework: 'torch',
            spec: {
              layers: [
                {
                  type: 'fleet.model_builder.layers.Concat',
                  name: 'Concat-0',
                  constructorArgs: {
                    dim: 1,
                  },
                  forwardArgs: {
                    xs: ['$sepal_length', '$sepal_width'],
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-1',
                  constructorArgs: {
                    in_features: 2,
                    out_features: 16,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$Concat-0',
                  },
                },
                {
                  type: 'torch.nn.ReLU',
                  name: 'ReLU-2',
                  constructorArgs: {
                    inplace: false,
                  },
                  forwardArgs: {
                    input: '$Linear-1',
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-3',
                  constructorArgs: {
                    in_features: 16,
                    out_features: 16,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$ReLU-2',
                  },
                },
                {
                  type: 'torch.nn.ReLU',
                  name: 'ReLU-4',
                  constructorArgs: {
                    inplace: false,
                  },
                  forwardArgs: {
                    input: '$Linear-3',
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-5',
                  constructorArgs: {
                    in_features: 16,
                    out_features: 1,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$ReLU-4',
                  },
                },
              ],
            },
            dataset: {
              name: 'IRIS_DATASET_NAME',
              strategy: 'forwardArgs',
              targetColumns: [
                {
                  name: 'large_petal_length',
                  dataType: {
                    domainKind: 'categorical',
                    classes: {
                      '0': 0,
                      '1': 1,
                    },
                  },
                  outModule: 'Linear-5',
                  lossFn: 'torch.nn.BCEWithLogitsLoss',
                  columnType: 'binary',
                },
              ],
              featureColumns: [
                {
                  name: 'sepal_length',
                  dataType: {
                    domainKind: 'numeric',
                    unit: 'mole',
                  },
                },
                {
                  name: 'sepal_width',
                  dataType: {
                    domainKind: 'numeric',
                    unit: 'mole',
                  },
                },
              ],
              featurizers: [],
              transforms: [],
            },
          },
          createdAt: '2023-08-16T15:58:16.738343+00:00',
          updatedAt: '2023-08-16T15:58:16.738343',
        },
        createdAt: '2023-08-28T19:48:56.902783+00:00',
        updatedAt: '2023-08-28T19:48:56.902783+00:00',
        createdById: 1,
        id: 126,
        mlflowId: null,
        stage: 'RUNNING',
        createdBy: {
          email: 'admin@mariner.trident.bio',
          isActive: true,
          isSuperuser: true,
          fullName: null,
          id: 1,
        },
        hyperparams: {
          learning_rate: 0.001,
          epochs: 299,
        },
        epochs: null,
        trainMetrics: null,
        valMetrics: null,
        testMetrics: null,
        history: null,
        stackTrace: null,
      },
      {
        experimentName: 'teste 3',
        modelVersionId: 1,
        modelVersion: {
          id: 1,
          modelId: 1,
          name: 'test model version',
          description: 'fwscttrs',
          mlflowVersion: '20',
          mlflowModelName:
            '1-SOME_MODEL_NAME-dfdada37-9a7f-4555-8670-5c697eedaf07',
          config: {
            name: 'test model version',
            framework: 'torch',
            spec: {
              layers: [
                {
                  type: 'fleet.model_builder.layers.Concat',
                  name: 'Concat-0',
                  constructorArgs: {
                    dim: 1,
                  },
                  forwardArgs: {
                    xs: ['$sepal_length', '$sepal_width'],
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-1',
                  constructorArgs: {
                    in_features: 2,
                    out_features: 16,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$Concat-0',
                  },
                },
                {
                  type: 'torch.nn.ReLU',
                  name: 'ReLU-2',
                  constructorArgs: {
                    inplace: false,
                  },
                  forwardArgs: {
                    input: '$Linear-1',
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-3',
                  constructorArgs: {
                    in_features: 16,
                    out_features: 16,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$ReLU-2',
                  },
                },
                {
                  type: 'torch.nn.ReLU',
                  name: 'ReLU-4',
                  constructorArgs: {
                    inplace: false,
                  },
                  forwardArgs: {
                    input: '$Linear-3',
                  },
                },
                {
                  type: 'torch.nn.Linear',
                  name: 'Linear-5',
                  constructorArgs: {
                    in_features: 16,
                    out_features: 1,
                    bias: true,
                  },
                  forwardArgs: {
                    input: '$ReLU-4',
                  },
                },
              ],
            },
            dataset: {
              name: 'IRIS_DATASET_NAME',
              strategy: 'forwardArgs',
              targetColumns: [
                {
                  name: 'large_petal_length',
                  dataType: {
                    domainKind: 'categorical',
                    classes: {
                      '0': 0,
                      '1': 1,
                    },
                  },
                  outModule: 'Linear-5',
                  lossFn: 'torch.nn.BCEWithLogitsLoss',
                  columnType: 'binary',
                },
              ],
              featureColumns: [
                {
                  name: 'sepal_length',
                  dataType: {
                    domainKind: 'numeric',
                    unit: 'mole',
                  },
                },
                {
                  name: 'sepal_width',
                  dataType: {
                    domainKind: 'numeric',
                    unit: 'mole',
                  },
                },
              ],
              featurizers: [],
              transforms: [],
            },
          },
          createdAt: '2023-08-16T15:58:16.738343+00:00',
          updatedAt: '2023-08-16T15:58:16.738343',
        },
        createdAt: '2023-08-28T19:14:37.790564+00:00',
        updatedAt: '2023-08-28T19:14:37.790564+00:00',
        createdById: 1,
        id: 125,
        mlflowId: null,
        stage: 'RUNNING',
        createdBy: {
          email: 'admin@mariner.trident.bio',
          isActive: true,
          isSuperuser: true,
          fullName: null,
          id: 1,
        },
        hyperparams: {},
        epochs: null,
        trainMetrics: null,
        valMetrics: null,
        testMetrics: null,
        history: null,
        stackTrace: null,
      },
    ];

    if (!value || !paginatedExperiments) return <div>error</div>;

    return (
      <Table
        columns={value.columns}
        rows={paginatedExperiments}
        rowKey={() => 'random'}
      />
    );
  },
};
