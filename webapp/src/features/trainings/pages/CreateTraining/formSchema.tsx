import * as yup from 'yup';
import { AnyObject } from 'yup/lib/types';

const requiredError = 'This field is required';

const validateAdvancedFields = (
  item: any,
  context: yup.TestContext<AnyObject>
) => {
  const currentFieldsAndValues = Object.entries(context.parent);
  for (const [key, value] of currentFieldsAndValues) {
    const advancedOptionContainValue = key.includes('early') && !!value;
    const currentItemWithoutValue = !item;
    if (advancedOptionContainValue && currentItemWithoutValue) {
      return false;
    }
  }
  return true;
};
const advancedOptionsRequiredError =
  'This field is required when advanced options is open';
