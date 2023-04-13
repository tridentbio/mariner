import { JSONSchemaType } from 'ajv';
import {
  DataTypeDomainKind,
  NewDataset,
  DataType,
} from 'app/types/domain/datasets';
import { DatasetForm } from './types';

export const DataTypeSchema: JSONSchemaType<DataType> = {
  type: 'object',
  required: [],
  errorMessage: {
    oneOf: 'Missing unit for numeric column',
  },
  oneOf: [
    {
      type: 'object',
      required: ['domainKind'],
      properties: {
        domainKind: {
          type: 'string',
          const: DataTypeDomainKind.Smiles,
        },
      },
    },
    {
      type: 'object',
      required: ['unit', 'domainKind'],
      properties: {
        domainKind: {
          type: 'string',
          const: DataTypeDomainKind.Numerical,
        },
        unit: {
          type: 'string',
        },
      },
    },
    {
      type: 'object',
      required: ['domainKind'],
      properties: {
        domainKind: {
          type: 'string',
          const: DataTypeDomainKind.Categorical,
        },
        classes: {
          type: 'object',
          patternProperties: {
            '^[A-Za-z0-9 ]*$': {
              type: 'integer',
            },
          },
        },
      },
    },
  ],
};
