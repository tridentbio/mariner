import { TablePreferences } from '@components/templates/Table/types';
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
        .lazy(
          (usersDict: {
            [userId: string | number]: { [preference: string]: any };
          }) => {
            if (!usersDict)
              throw new yup.ValidationError(
                'User preference schema must not be empty'
              );

            const userPreferenceSchema: { [preference: string]: any } = {};

            Object.entries(usersDict).forEach(([userId, preferenceData]) => {
              userPreferenceSchema[userId] = yup.object({
                tables: ((): yup.ObjectSchema<any> => {
                  if (!preferenceData?.tables)
                    throw new yup.ValidationError(
                      'Table preference schema must not be empty'
                    );

                  const tablePreferencesSchema = notEmptyObject()
                    .shape({
                      columns: yup
                        .array()
                        .of(
                          yup
                            .object()
                            .shape({
                              name: yup.string().nullable(),
                            })
                            .required()
                        )
                        .min(1),
                    })
                    .noUnknown();

                  const tablesPreferencesSchema = Object.keys(
                    preferenceData.tables as {
                      [tableId: string]: TablePreferences;
                    }
                  ).reduce<{ [tableId: string]: yup.ObjectSchema<any> }>(
                    (acc, tableId) => {
                      acc[tableId] = tablePreferencesSchema;
                      return acc;
                    },
                    {}
                  );

                  return notEmptyObject().shape(tablesPreferencesSchema);
                })(),
              });
            });

            return notEmptyObject().shape(userPreferenceSchema).noUnknown();
          }
        )
        .validateSync(value);
    },
  },
};

export const fetchLocalStorage = <T extends object>(key: ELocalStorage) => {
  try {
    const data = localStorage.getItem(key);

    if (data) {
      const parsedPreferences: T = JSON.parse(data);

      try {
        storageSchemas[key] && storageSchemas[key]?.validate(parsedPreferences);
      } catch (error: any) {
        throw new yup.ValidationError(error);
      }

      return parsedPreferences;
    }

    return null;
  } catch (error) {
    error instanceof yup.ValidationError
      ? // eslint-disable-next-line no-console
        console.error('Storage data has a invalid schema', error)
      : // eslint-disable-next-line no-console
        console.error(error);

    return null;
  }
};
