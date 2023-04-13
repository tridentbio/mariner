import { api } from '../api';
export const addTagTypes = ['events'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      getEventsReport: build.query<
        GetEventsReportApiResponse,
        GetEventsReportApiArg
      >({
        query: () => ({ url: `/api/v1/events/report` }),
        providesTags: ['events'],
      }),
      postEventsRead: build.mutation<
        PostEventsReadApiResponse,
        PostEventsReadApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/events/read`,
          method: 'POST',
          body: queryArg.readRequest,
        }),
        invalidatesTags: ['events'],
      }),
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as enhancedApi };
export type GetEventsReportApiResponse =
  /** status 200 Successful Response */ EventsbySource[];
export type GetEventsReportApiArg = void;
export type PostEventsReadApiResponse =
  /** status 200 Successful Response */ EventsReadResponse;
export type PostEventsReadApiArg = {
  readRequest: ReadRequest;
};
export type Event = {
  id: number;
  userId?: number;
  source: 'training:completed' | 'changelog';
  timestamp: string;
  payload: object;
  url?: string;
};
export type EventsbySource = {
  source: 'training:completed' | 'changelog';
  total: number;
  message: string;
  events: Event[];
};
export type EventsReadResponse = {
  total: number;
};
export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
};
export type HttpValidationError = {
  detail?: ValidationError[];
};
export type ReadRequest = {
  eventIds: number[];
};
export const {
  useGetEventsReportQuery,
  useLazyGetEventsReportQuery,
  usePostEventsReadMutation,
} = injectedRtkApi;
