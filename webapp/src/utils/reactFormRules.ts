import { sum } from '@utils';

export const required = {
  required: {
    value: true,
    message: 'This field is required',
  },
};

export const minLength = (value: number) => ({
  minLength: {
    value,
    message: 'The field must not be empty',
  },
});
export const splitString = {
  validate: {
    value: (val: string) => {
      if (typeof val === 'string') {
        const parts = val.split('-');
        if (parts.length !== 3) return false;
        const intParst = parts.map((s) => parseInt(s));
        if (intParst.includes(NaN)) return false;
        if (sum(intParst) !== 100) return false;
        return true;
      }
      return false;
    },
    message: 'Should be a valid split division, e.g.: 60-20-20, 80-10-10',
  },
};
