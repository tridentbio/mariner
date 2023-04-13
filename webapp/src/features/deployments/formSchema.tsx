import * as yup from 'yup';
import { EShareStrategies } from './types';

const requiredError = 'This field is required';

export const deploymentFormSchema = yup
  .object({
    name: yup.string().required(requiredError),
    readme: yup.string(),
    modelVersionId: yup.number().positive().integer().required(requiredError),
    shareStrategy: yup.mixed().oneOf(Object.values(EShareStrategies)),
    usersIdAllowed: yup
      .array()
      .of(yup.number().positive().integer().required(requiredError))
      .when('shareStrategy', {
        is: EShareStrategies.PUBLIC,
        then: (field) => field.nullable().transform(() => null),
      }),
    organizationsAllowed: yup
      .array()
      .of(yup.string().required(requiredError))
      .when('shareStrategy', {
        is: EShareStrategies.PUBLIC,
        then: (field) => field.nullable().transform(() => null),
      }),
  })
  .required();
