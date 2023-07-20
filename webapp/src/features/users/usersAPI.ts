import api from '../../app/api';

export interface User {
  email: string;
  id: number;
  full_name: string;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface Provider {
  logo_url: string;
  id: string;
  name: string;
}

export const login = async (
  username: string,
  password: string
): Promise<TokenResponse> => {
  const data = new FormData();

  data.set('username', username);
  data.set('password', password);

  return api.post('api/v1/login/access-token', data).then((res) => {
    return res.data;
  });
};

export const getMe = async (): Promise<User> => {
  return api.get('api/v1/users/me').then((res) => res.data);
};

export const getProviders = async (): Promise<Provider[]> => {
  return api.get('api/v1/oauth-providers').then((res) => res.data);
};
