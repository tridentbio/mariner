import { Meta, StoryObj } from '@storybook/react';
import ProcessingModal from '.';
export default {
  title: 'Components/Processing Modal',
  component: ProcessingModal,
  args: {
    processing: true,
    type: 'Saving',
  },
  decorators: [
    (Story) => {
      return <div style={{ width: '100vw', height: '100vh' }}>{Story()}</div>;
    },
  ],
} as Meta;

export const Saving: StoryObj = {};
export const Fetching: StoryObj = {
  args: {
    type: 'Fetching',
  },
};
export const Checking: StoryObj = {
  args: {
    type: 'Checking',
  },
};
