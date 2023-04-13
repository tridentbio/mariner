import { api } from './api';
import { User } from 'features/users/usersAPI';

type UsersQuery = {
  skip?: number;
  limit?: number;
};

export const usersApiRtk = api
  .enhanceEndpoints({ addTagTypes: ['users'] })
  .injectEndpoints({
    endpoints: (builder) => ({
      getUsers: builder.query<User[], UsersQuery>({
        query: (params) => ({ url: 'api/v1/users', params }),
        providesTags: (_result, _error, arg) => [
          { type: 'users', skip: arg.skip || 0, limit: arg.limit || 100 },
        ],
      }),
    }),
  });
