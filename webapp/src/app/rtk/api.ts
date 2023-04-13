import { SerializedError } from '@reduxjs/toolkit';
import {
  createApi,
  fetchBaseQuery,
  FetchBaseQueryError,
} from '@reduxjs/toolkit/query/react';
import { TOKEN } from '../local-storage';

export const isFetchBaseQueryError = (
  error: SerializedError | unknown | FetchBaseQueryError | undefined
): error is FetchBaseQueryError => {
  return !!error && typeof error === 'object' && 'status' in error;
};
interface ApiError {
  data: {
    detail:
      | string
      | { loc: any; msg: string; type: any }
      | { loc: any; msg: string; type: any }[];
  };
  status: number;
}
export const messageApiError = (apiError: ApiError) => {
  const { detail } = apiError.data;
  if (typeof detail === 'string') return detail;
  else if ('msg' in detail) {
    return detail.msg;
  } else {
    return 'Internal Error';
  }
};
export const isApiError = (error: any): error is ApiError => {
  return (
    typeof error === 'object' &&
    error !== null &&
    'data' in error &&
    'detail' in error.data
  );
};

const baseQuery = fetchBaseQuery({
  baseUrl: `${import.meta.env.VITE_API_BASE_URL}`,
  prepareHeaders: (headers) => {
    const storage = localStorage.getItem(TOKEN);
    if (storage) {
      const token = JSON.parse(storage).access_token;
      if (token) {
        headers.set('Authorization', `Bearer ${token}`);
      }
    }
    return headers;
  },
});

export const api = createApi({
  reducerPath: 'api',
  baseQuery,
  endpoints: () => ({}),
});
