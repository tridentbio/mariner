import api from 'app/api';

export interface Unit {
  name: string;
  latex?: string;
}
export const fetchUnits = async (query: string = ''): Promise<Unit[]> =>
  api
    .get('api/v1/units/', {
      params: { q: query },
    })
    .then((res) => res.data);

export const isUnitValid = async (query: string): Promise<boolean> =>
  api
    .get('api/v1/units/valid', { params: { q: query } })
    .then((res) => res.status === 200);
