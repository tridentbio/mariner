import { FieldError } from 'react-hook-form';
import { PreprocessingStepSelectGetErrorFn } from './PreprocessingStepSelect';
import { SimpleColumnConfig, StepValue } from './types';
import { ColumnConfig } from '@app/rtk/generated/models';

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

export const getColumnConfigTestId = (
  column: SimpleColumnConfig | ColumnConfig
) => `${column!.name!}-${column!.dataType!.domainKind!}`;

export const getStepValueLabelData = (stepType: StepValue['type']) => {
  if (stepType) {
    const parts = stepType.split('.');
    const lib = parts[0];
    const class_ = parts.at(-1) as string;

    return { class: class_, lib };
  }
};
