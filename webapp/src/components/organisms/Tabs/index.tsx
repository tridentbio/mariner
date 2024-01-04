import { ReactNode, SyntheticEvent, useState } from 'react';
import { Tabs, Box, Tab, SxProps, Theme } from '@mui/material';
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
  boxProps?: {
    sx: SxProps<Theme>;
  };
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  if (value === index)
    return (
      <Box
        role="tabpanel"
        hidden={value !== index}
        id={`simple-tabpanel-${index}`}
        aria-labelledby={`simple-tab-${index}`}
        {...other}
        {...props.boxProps}
      >
        {children}
      </Box>
    );

  return null;
}

function a11yProps(index: number) {
  return {
    id: `simple-tab-${index}`,
    'aria-controls': `simple-tabpanel-${index}`,
  };
}

export interface AppTabsProps {
  tabs: {
    label: string;
    panel: ReactNode;
  }[];
  initialTab?: number;
  boxProps?: TabPanelProps['boxProps'];
  onChange?: (tabValue: number) => void;
}

export default function AppTabs(props: AppTabsProps) {
  const [value, setValue] = useState(props.initialTab || 0);

  const handleChange = (_event: SyntheticEvent, newValue: number) => {
    setValue(newValue);

    props.onChange && props.onChange(newValue);
  };

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={value} onChange={handleChange}>
          {props.tabs.map((tab, index) => (
            <Tab key={tab.label} label={tab.label} {...a11yProps(index)} />
          ))}
        </Tabs>
      </Box>
      {props.tabs.map((tab, index) => (
        <TabPanel
          key={index}
          value={value}
          index={index}
          boxProps={props.boxProps}
        >
          {tab.panel}
        </TabPanel>
      ))}
    </Box>
  );
}
