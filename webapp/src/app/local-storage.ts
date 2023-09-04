import * as yup from 'yup';

export enum ELocalStorage {
  PREFERENCES = 'preferences',
  TOKEN = 'app-token',
}

export const storageSchemas: {
  [storageKey in ELocalStorage]?: { validate: (value: any) => void };
} = {
  preferences: {
    validate(value) {
      const notEmptyObject = (message = 'Object must not be empty') =>
        yup.object().test({
          name: 'notEmpty',
          message,
          test: (value) => {
            return Object.keys(value).length > 0;
          },
        });

      return yup
        .object({
          tables: yup.lazy((value) => {
            if (!value) return yup.mixed();

            const tablePreferencesSchema = notEmptyObject()
              .shape({
                columns: yup.array().of(
                  yup.object({
                    field: yup.string().nullable(),
                  })
                ),
              })
              .required();

            const tables = Object.keys(value).reduce((acc, tableId) => {
              acc[tableId] = tablePreferencesSchema;
              return acc;
            }, {} as { [key: string]: any });

            return notEmptyObject().shape(tables);
          }),
        })
        .validateSync(value);
    },
  },
};
