import { createAsyncThunk, createSlice, current } from '@reduxjs/toolkit';
import { Status } from 'app/api';
import * as notificationsAPI from './notificationsAPI';

interface NotificationState {
  notifications: notificationsAPI.MarinerNotification[];
  fetchingNotifications: Status;
  updatingEventIds: number[];
}

const initialState: NotificationState = {
  notifications: [],
  fetchingNotifications: 'idle',
  updatingEventIds: [],
};

export const fetchNotificaitions = createAsyncThunk(
  'notifications/fetchNotifications',
  notificationsAPI.getNotifications
);

export const updateEventsAsRead = createAsyncThunk(
  'notifications/updateNotificationAsRead',
  notificationsAPI.setNotfiicationsRead
);

const notificationSlice = createSlice({
  name: 'notifications',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchNotificaitions.fulfilled, (state, action) => {
        state.notifications = action.payload;
        state.fetchingNotifications = 'idle';
      })
      .addCase(fetchNotificaitions.pending, (state) => {
        state.fetchingNotifications = 'loading';
      })
      .addCase(fetchNotificaitions.rejected, (state) => {
        state.fetchingNotifications = 'failed';
      });
    builder
      .addCase(updateEventsAsRead.pending, (state, action) => {
        state.updatingEventIds = action.meta.arg;
      })
      .addCase(updateEventsAsRead.fulfilled, (state, action) => {
        state.notifications = state.notifications.map((notif) => {
          return {
            ...notif,
            events: notif.events.filter(
              (event) => !state.updatingEventIds.includes(event.id)
            ),
            total: notif.total - action.meta.arg.length,
          };
        });
        state.updatingEventIds = [];
      })
      .addCase(updateEventsAsRead.rejected, (state) => {
        state.updatingEventIds = [];
      });
  },
});

export default notificationSlice.reducer;
