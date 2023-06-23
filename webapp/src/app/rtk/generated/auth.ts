import { api } from '../api';
export const addTagTypes = ['login', 'oauth'] as const;
const injectedRtkApi = api
  .enhanceEndpoints({
    addTagTypes,
  })
  .injectEndpoints({
    endpoints: (build) => ({
      loginAccessToken: build.mutation<
        LoginAccessTokenApiResponse,
        LoginAccessTokenApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/login/access-token`,
          method: 'POST',
          body: queryArg.bodyLoginAccessTokenApiV1LoginAccessTokenPost,
        }),
        invalidatesTags: ['login'],
      }),
      getOauthProviderRedirect: build.query<
        GetOauthProviderRedirectApiResponse,
        GetOauthProviderRedirectApiArg
      >({
        query: (queryArg) => ({
          url: `/api/v1/oauth`,
          params: { provider: queryArg.provider },
        }),
        providesTags: ['oauth'],
      }),
    }),
    overrideExisting: false,
  });
export { injectedRtkApi as enhancedApi };
export type LoginAccessTokenApiResponse =
  /** status 200 Successful Response */ Token;
export type LoginAccessTokenApiArg = {
  bodyLoginAccessTokenApiV1LoginAccessTokenPost: BodyLoginAccessTokenApiV1LoginAccessTokenPost;
};
export type GetOauthProviderRedirectApiResponse =
  /** status 200 Successful Response */ any;
export type GetOauthProviderRedirectApiArg = {
  provider: string;
};
export type Token = {
  access_token: string;
  token_type: string;
};
export type ValidationError = {
  loc: (string | number)[];
  msg: string;
  type: string;
};
export type HttpValidationError = {
  detail?: ValidationError[];
};
export type BodyLoginAccessTokenApiV1LoginAccessTokenPost = {
  grant_type?: string;
  username: string;
  password: string;
  scope?: string;
  client_id?: string;
  client_secret?: string;
};
export const {
  useLoginAccessTokenMutation,
  useGetOauthProviderRedirectQuery,
  useLazyGetOauthProviderRedirectQuery,
} = injectedRtkApi;
