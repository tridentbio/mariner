import axios from 'axios';
import { makeHttpError, Status as HTTPStatus } from 'utils/http';
import { TOKEN } from './local-storage';

export type Status = 'idle' | 'loading' | 'failed';

export interface Paginated<T> {
  total: number;
  data: T[];
}

const api = axios.create({
  validateStatus: (status) => status === 200,
  baseURL: import.meta.env.VITE_API_BASE_URL,
});

api.interceptors.response.use(async (res) => {
  if (res.status !== HTTPStatus.OK) throw makeHttpError(res);

  return res;
});

api.interceptors.request.use((config) => {
  const storage = localStorage.getItem(TOKEN);

  if (storage) {
    const token = JSON.parse(storage).access_token;

    if (!config || !config.headers) return;
    config.headers.Authorization = `Bearer ${token}`;
  }

  return config;
});

export default api;
