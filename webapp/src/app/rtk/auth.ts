import { makeForm } from 'utils';
import { api } from './api';

export interface User {
  email: string;
  id: number;
  full_name: string;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
}

export const authApi = api.injectEndpoints({
  endpoints: (builder) => {
    return {
      login: builder.mutation<
        TokenResponse,
        { username: string; password: string }
      >({
        query: (payload) => ({
          url: 'api/v1/login/access-token',
          method: 'POST',
          body: makeForm(payload),
        }),
      }),
      getMe: builder.query<User, void>({
        query: () => `api/v1/users/me`,
      }),
    };
  },
});
