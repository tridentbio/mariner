import { randomLowerCase } from '@utils';
import { DATASET_NAME } from '../constants';
import { DatasetFormData } from './create';

export const zincDatasetFixture: DatasetFormData = {
  name: DATASET_NAME,
  description: randomLowerCase(24),
  file: 'zinc.csv',
  split: '80-10-10',
  splitType: 'Random',
  splitColumn: 'smiles',
  descriptions: [
    {
      pattern: 'smiles',
      dataType: {
        domainKind: 'SMILES',
      },
      description: 'A smile column',
    },
    {
      pattern: 'mwt_group',
      dataType: {
        domainKind: 'Categorical',
      },
      description: 'A categorical column',
    },
    {
      pattern: 'tpsa',
      dataType: {
        domainKind: 'Numeric',
        unit: 'mole',
      },
      description: 'Another numerical column',
    },
    {
      pattern: 'zinc_id',
      dataType: {
        domainKind: 'String',
      },
      description: '--',
    },
    {
      pattern: 'mwt',
      dataType: {
        domainKind: 'Numeric',
        unit: 'mole',
      },
      description: 'A numerical column',
    },
  ],
};

export const createIrisDatasetFormData = (): DatasetFormData => {
  return {
    name: randomLowerCase(8),
    description: randomLowerCase(24),
    file: 'data/csv/iris.csv',
    split: '60-20-20',
    splitType: 'Random',
    splitColumn: 'species',
    descriptions: [
      {
        pattern: 'sepal_length',
        dataType: { domainKind: 'Numeric', unit: 'mole' },
        description: randomLowerCase(12),
      },
      {
        pattern: 'sepal_width',
        dataType: { domainKind: 'Numeric', unit: 'mole' },
        description: randomLowerCase(12),
      },
      {
        pattern: 'petal_length',
        dataType: { domainKind: 'Numeric', unit: 'mole' },
        description: randomLowerCase(12),
      },
      {
        pattern: 'petal_width',
        dataType: { domainKind: 'Numeric', unit: 'mole' },
        description: randomLowerCase(12),
      },
      {
        pattern: 'species',
        dataType: {
          domainKind: 'Categorical',
          classes: { '0': 0, '1': 1, '2': 2 },
        },
        description: randomLowerCase(12),
      },
      {
        pattern: 'large_petal_length',
        dataType: { domainKind: 'Categorical', classes: { '0': 0, '1': 1 } },
        description: randomLowerCase(12),
      },
      {
        pattern: 'large_petal_width',
        dataType: { domainKind: 'Categorical', classes: { '0': 0, '1': 1 } },
        description: randomLowerCase(12),
      },
    ],
  };
};

export const createRandomDatasetFormData = (): DatasetFormData => {
  return {
    name: randomLowerCase(8),
    description: randomLowerCase(24),
    file: 'zinc.csv',
    split: '80-10-10',
    splitType: 'Random',
    splitColumn: 'smiles',
    descriptions: [
      {
        pattern: 'smiles',
        dataType: {
          domainKind: 'SMILES',
        },
        description: 'A smile column',
      },
      {
        pattern: 'mwt_group',
        dataType: {
          domainKind: 'Categorical',
          classes: { mwt_big: 0, mwt_small: 1 },
        },

        description: 'A categorical column',
      },
      {
        pattern: 'tpsa',
        dataType: {
          domainKind: 'Numeric',
          unit: 'mole',
        },
        description: 'Another numerical column',
      },
      {
        pattern: 'zinc_id',
        dataType: {
          domainKind: 'String',
        },
        description: '--',
      },
      {
        pattern: 'mwt',
        dataType: {
          domainKind: 'Numeric',
          unit: 'mole',
        },
        description: 'A numerical column',
      },
    ],
  };
};

export const getDatasetsMock = {
  data: [
    {
      name: 'New Data',
      description: 'teste',
      rows: 501,
      columns: 7,
      bytes: 46513,
      dataUrl: 'datasets/d41d8cd98f00b204e9800998ecf8427e.csv',
      splitTarget: '80-10-10',
      splitActual: null,
      splitType: 'random',
      createdAt: '2022-11-15T16:04:15.295773+00:00',
      updatedAt: '2022-11-15T16:04:15.295777+00:00',
      createdById: 1,
      columnsMetadata: [
        {
          dataType: {
            domainKind: 'string',
          },
          description: '--',
          pattern: 'zinc_id',
          datasetId: 22,
        },
        {
          dataType: {
            domainKind: 'smiles',
          },
          description: 'A smile column',
          pattern: 'smiles',
          datasetId: 22,
        },
        {
          dataType: {
            domainKind: 'numeric',
            unit: 'mole',
          },
          description: 'A numerical column',
          pattern: 'mwt',
          datasetId: 22,
        },
        {
          dataType: {
            domainKind: 'numeric',
            unit: 'mole',
          },
          description: 'Another numerical column',
          pattern: 'tpsa',
          datasetId: 22,
        },
        {
          dataType: {
            domainKind: 'categorical',
            classes: {
              mwt_big: 0,
              mwt_small: 1,
            },
          },
          description: 'A categorical column',
          pattern: 'mwt_group',
          datasetId: 22,
        },
        {
          dataType: {
            domainKind: 'string',
          },
          description: '',
          pattern: 'Unnamed: 0.1',
          datasetId: 22,
        },
        {
          dataType: {
            domainKind: 'string',
          },
          description: '',
          pattern: 'Unnamed: 0',
          datasetId: 22,
        },
      ],
      id: 22,
    },
  ],
  total: 1,
};
