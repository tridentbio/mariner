import * as yup from 'yup';
import { DataTypeGuard } from '@app/types/domain/datasets';
import { SimpleColumnConfig } from './types';
import { TypeIdentifier } from '@hooks/useModelOptions';
import { MixedSchema } from 'yup/lib/mixed';
import { AnyObject } from 'yup/lib/types';

const requiredError = 'This field is required';

export const preprocessingStepSchema = yup.object({
  type: yup.string().required(requiredError),
  constructorArgs: yup.lazy((args: { [key: string]: object }) => {
    const buildedSchema: { [key: string]: yup.AnySchema } = {};

    if (args) {
      Object.entries(args).forEach(([arg, argData]) => {
        const argSchema: { [key: string]: yup.AnySchema } = {};

        argData &&
          Object.keys(argData).forEach((key) => {
            switch (key) {
              case 'required':
                argSchema[key] = yup.boolean();
                break;
              case 'type':
                argSchema[key] = yup.mixed<TypeIdentifier>().required();
                break;
              case 'default': {
                argSchema[key] = yup
                  .mixed<string | number | boolean>()
                  .when('required', {
                    is: true,
                    then: (field) => field.required(),
                  })
                  .when(
                    ['required', 'type'],
                    // @ts-ignore
                    (
                      required: boolean,
                      type: TypeIdentifier,
                      field: MixedSchema<
                        string | number | boolean,
                        AnyObject,
                        any
                      >
                    ) => {
                      return required && type === 'boolean'
                        ? field.required().test((value) => !!value)
                        : field.required();
                    }
                  );
                break;
              }
            }
          });

        buildedSchema[arg] = yup.object(argSchema).required();
      });
    }

    return yup.object(buildedSchema);
  }),
});

export const columnSchema = yup.object({
  name: yup.string().required(requiredError),
  dataType: yup.object({
    domainKind: yup.string().required(requiredError),
    unit: yup.string(),
  }),
  featurizers: yup
    .array()
    .of(preprocessingStepSchema)
    .when('dataType.domainKind', {
      is: (domainKind: SimpleColumnConfig['dataType']['domainKind']) =>
        !DataTypeGuard.isNumericalOrQuantity(domainKind),
      then: (field) => field.min(1).required(),
    }),
  transforms: yup.array().of(preprocessingStepSchema),
});

export const dataPreprocessingFormSchema = yup
  .object({
    featureColumns: yup.array().of(columnSchema),
    targetColumns: yup.array().of(columnSchema),
  })
  .required();
