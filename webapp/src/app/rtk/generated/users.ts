import { api } from '../api';
export const addTagTypes = ['users'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      readUsers: build.query<ReadUsersApiResponse, ReadUsersApiArg>({
        query: (queryArg) => ({
          url: `/api/v1/users/`,
          params: { skip: queryArg.skip, limit: queryArg.limit },
        }),
        providesTags: ['users'],
      }),
      readUserMe: build.query<ReadUserMeApiResponse, ReadUserMeApiArg>({
        query: () => ({ url: `/api/v1/users/me` }),
        providesTags: ['users'],
      }),
      updateUserMe: build.mutation<UpdateUserMeApiResponse, UpdateUserMeApiArg>(
        {
          query: (queryArg) => ({
            url: `/api/v1/users/me`,
            method: 'PUT',
            params: {
              email: queryArg.email,
              isActive: queryArg.isActive,
              isSuperuser: queryArg.isSuperuser,
              fullName: queryArg.fullName,
              password: queryArg.password,
            },
          }),
          invalidatesTags: ['users'],
        }
      ),
      updateUser: build.mutation<UpdateUserApiResponse, UpdateUserApiArg>({
        query: (queryArg) => ({
          url: `/api/v1/users/${queryArg.userId}`,
          method: 'PUT',
          body: queryArg.userUpdate,
        }),
        invalidatesTags: ['users'],
      }),
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as enhancedApi };
export type ReadUsersApiResponse = /** status 200 Successful Response */ User[];
export type ReadUsersApiArg = {
  skip?: number;
  limit?: number;
};
export type ReadUserMeApiResponse = /** status 200 Successful Response */ User;
export type ReadUserMeApiArg = void;
export type UpdateUserMeApiResponse =
  /** status 200 Successful Response */ User;
export type UpdateUserMeApiArg = {
  email?: string;
  isActive?: boolean;
  isSuperuser?: boolean;
  fullName?: string;
  password?: string;
};
export type UpdateUserApiResponse = /** status 200 Successful Response */ User;
export type UpdateUserApiArg = {
  userId: number;
  userUpdate: UserUpdate;
};
export type User = {
  email?: string;
  isActive?: boolean;
  isSuperuser?: boolean;
  fullName?: string;
  id?: number;
};
export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
};
export type HttpValidationError = {
  detail?: ValidationError[];
};
export type UserUpdate = {
  email?: string;
  isActive?: boolean;
  isSuperuser?: boolean;
  fullName?: string;
  password?: string;
};
export const {
  useReadUsersQuery,
  useLazyReadUsersQuery,
  useReadUserMeQuery,
  useLazyReadUserMeQuery,
  useUpdateUserMeMutation,
  useUpdateUserMutation,
} = injectedRtkApi;
