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
    let invalid = false;
    const fieldError = getFormFieldError();

    switch (type) {
      case 'type':
        if (fieldError) invalid = !!fieldError.type;
        break;
      default: {
        if (params?.config.required && !value) return true;

        if (fieldError) {
          const constructorArgsError = fieldError.constructorArgs;
          if (constructorArgsError)
            invalid = !!constructorArgsError[params?.key as string];
        }
      }
    }

    return invalid;
  };
};
