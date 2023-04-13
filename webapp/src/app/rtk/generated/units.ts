import { api } from '../api';
export const addTagTypes = ['units'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      getUnits: build.query<GetUnitsApiResponse, GetUnitsApiArg>({
        query: (queryArg) => ({
          url: `/api/v1/units/`,
          params: { q: queryArg.q },
        }),
        providesTags: ['units'],
      }),
      getIsUnitValid: build.query<
        GetIsUnitValidApiResponse,
        GetIsUnitValidApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/units/valid`,
          params: { q: queryArg.q },
        }),
        providesTags: ['units'],
      }),
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as enhancedApi };
export type GetUnitsApiResponse = /** status 200 Successful Response */ Unit[];
export type GetUnitsApiArg = {
  q: string;
};
export type GetIsUnitValidApiResponse =
  /** status 200 Successful Response */ Unit;
export type GetIsUnitValidApiArg = {
  q: string;
};
export type Unit = {
  name: string;
  latex: string;
};
export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
};
export type HttpValidationError = {
  detail?: ValidationError[];
};
export const {
  useGetUnitsQuery,
  useLazyGetUnitsQuery,
  useGetIsUnitValidQuery,
  useLazyGetIsUnitValidQuery,
} = injectedRtkApi;
