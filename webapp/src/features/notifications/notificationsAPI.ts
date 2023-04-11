import api from '../../app/api';

interface CommonEvent {
  id: number;
  url: string;
  timestamp: Date;
}

export interface TrainingEvent extends CommonEvent {
  source: 'training:completed';
  payload: {
    id: number;
    experiment_name: string;
  };
}

export interface ChangelogEvent extends CommonEvent {
  source: 'changelog';
  payload: {
    version: string;
    date: Date;
    changes: {
      type:
        | 'Added'
        | 'Fixed'
        | 'Updated'
        | 'Deprecated'
        | 'Removed'
        | 'Security';
      message: string;
    }[];
  };
}

export type MarinerEvent = TrainingEvent | ChangelogEvent;

export interface MarinerNotification {
  source: MarinerEvent['source'];
  message: string;
  total: number;
  events: MarinerEvent[];
}

export const getNotifications = async (): Promise<MarinerNotification[]> =>
  api.get('/api/v1/events/report').then((res) => res.data);

export const setNotfiicationsRead = async (
  eventIds: number[]
): Promise<{ total: number }> =>
  api.post('/api/v1/events/read', { eventIds }).then((res) => res.data);
