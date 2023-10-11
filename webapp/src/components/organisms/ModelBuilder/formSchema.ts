import { DataTypeGuard } from '@app/types/domain/datasets';
import * as yup from 'yup';
import { SimpleColumnConfig } from './types';

const requiredError = 'This field is required';

export const preprocessingStepSchema = yup.object({
  type: yup.string().required(requiredError),
  constructorArgs: yup.object().nullable(),
});

const columnConfigSchema = yup.object({
  name: yup.string().required(requiredError),
  dataType: yup.object({
    domainKind: yup.string().required(requiredError),
    unit: yup.string(),
  }),
});

export const simpleColumnSchema = columnConfigSchema.shape({
  featurizers: yup
    .array()
    .of(preprocessingStepSchema)
    .when('dataType', {
      is: (dataType: SimpleColumnConfig['dataType']) =>
        !DataTypeGuard.isNumericalOrQuantity(dataType),
      then: (field) => field.min(1).required(),
    }),
  transforms: yup.array().of(preprocessingStepSchema),
});

export const sklearnDatasetSchema = yup.object({
  name: yup.string().required('Dataset is required'),
  featureColumns: yup
    .array()
    .required('The feature columns are required')
    .min(1, 'The feature columns must not be empty')
    .of(simpleColumnSchema),
  targetColumns: yup
    .array()
    .required()
    .min(1, 'The target columns must not be empty')
    .of(simpleColumnSchema),
});

export const torchDatasetSchema = yup.object({
  name: yup.string().required('Dataset is required'),
  featureColumns: yup
    .array()
    .required('The feature columns are required')
    .of(columnConfigSchema)
    .min(1, 'The feature columns must not be empty'),
  targetColumns: yup
    .array()
    .required()
    .of(columnConfigSchema)
    .min(1, 'The target columns must not be empty'),
  featurizers: yup.array().required(),
  transforms: yup.array().required(),
});
