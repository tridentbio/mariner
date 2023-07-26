import { User } from 'features/users/usersAPI';
import { generatedUsersApi } from './generated/users';

type UsersQuery = {
  skip?: number;
  limit?: number;
};

export const usersApiRtk = generatedUsersApi
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
