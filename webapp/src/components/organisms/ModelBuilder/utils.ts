import { FieldError } from 'react-hook-form';
import { PreprocessingStepSelectGetErrorFn } from './PreprocessingStepSelect';

export type StepFormFieldError = {
  type: FieldError;
  constructorArgs: { [key: string]: FieldError };
};

export const getStepSelectError = (
  getFormFieldError: () => StepFormFieldError | undefined
): PreprocessingStepSelectGetErrorFn => {
  return (type, value, params) => {
    const formFieldError = getFormFieldError();

    if (type == 'constructorArgs' && params?.config.required && !value)
      return true;

    if (formFieldError) {
      switch (type) {
        case 'type':
          return !!formFieldError.type;
        case 'constructorArgs':
          return !!formFieldError.constructorArgs?.[params?.key as string];
      }
    }
    return false;
  };
};
