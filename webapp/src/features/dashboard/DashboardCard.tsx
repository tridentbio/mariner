import { FC, ReactNode, useMemo } from 'react';
import { Typography, Badge, Box } from '@mui/material';
import { MarinerNotification } from 'features/notifications/notificationsAPI';
import Card from 'components/organisms/Card';
import { useNavigate } from 'react-router-dom';

export interface DashboardCardProps {
  title: string;
  description: string;
  url: string;
  notifications: MarinerNotification[];
  children?: ReactNode;
}

const DashboardCard: FC<DashboardCardProps> = ({
  url,
  children,
  title,
  description,
  notifications,
}) => {
  const totalNotifications = notifications.length ? notifications[0].total : 0;
  const navigate = useNavigate();

  return (
    <Box
      sx={{ cursor: 'pointer' }}
      onClick={() => {
        navigate(url);
        return false;
      }}
    >
      <Card
        sx={{
          boxShadow: '0px 3px 8px rgba(0,0,0,0.23)',
          transition: 'box-shadow 0.3s',
          '&:hover': {
            boxShadow: '0px 3px 17px rgba(0,0,0,0.23)',
          },
        }}
        title={
          !!totalNotifications ? (
            <Badge badgeContent={totalNotifications} color="primary">
              <Typography fontSize="h5.fontSize" color="primary">
                {title}
              </Typography>
            </Badge>
          ) : (
            <Typography fontSize="h5.fontSize" color="primary">
              {title}
            </Typography>
          )
        }
      >
        <Typography variant="body1">{description}</Typography>
        {children}
      </Card>
    </Box>
  );
};

export default DashboardCard;
