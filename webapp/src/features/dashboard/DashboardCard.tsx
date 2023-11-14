import { Badge, Box, Divider, Typography, useTheme } from '@mui/material';
import Card from 'components/organisms/Card';
import { MarinerNotification } from 'features/notifications/notificationsAPI';
import { FC, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';

export interface DashboardCardProps {
  title: string;
  description: string;
  url: string;
  notifications: MarinerNotification[];
  children?: ReactNode;
  icon?: ReactNode;
}

const DashboardCard: FC<DashboardCardProps> = ({
  url,
  children,
  title,
  description,
  notifications,
  icon,
}) => {
  const totalNotifications = notifications.length ? notifications[0].total : 0;
  const navigate = useNavigate();
  const theme = useTheme();

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
          p: 'none',
          boxShadow: '0px 3px 8px rgba(0,0,0,0.15)',
          borderRadius: '10px',
          transition: 'all 0.3s',
          overflow: 'initial',
          '&:hover': {
            boxShadow: '0px 3px 17px rgba(0,0,0,0.15)',
            transform: 'translateY(-4px)',
          },
          '.MuiCardContent-root': {
            p: 'none',
          },
          '.MuiCardContent-root:last-child': {
            p: 'none',
          },
        }}
        title={
          <>
            <Badge
              badgeContent={totalNotifications ?? null}
              sx={{
                display: 'flex',
                '.MuiBadge-badge': {
                  top: -15,
                  right: -10,
                },
              }}
              color="primary"
            >
              <Box
                display="flex"
                alignItems="center"
                sx={{
                  mb: 2,
                  width: '100%',
                }}
              >
                {icon ? (
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '1.5rem',
                      p: 1.2,
                      mr: 1.5,
                      borderRadius: '12px',
                      backgroundColor: theme.palette.primary.main,
                    }}
                  >
                    {icon}
                  </Box>
                ) : null}
                <Typography fontSize="h5.fontSize" color="primary">
                  {title}
                </Typography>
              </Box>
            </Badge>
            <Divider sx={{ mb: 2 }} />
          </>
        }
      >
        <Typography variant="body1">{description}</Typography>
        {children}
      </Card>
    </Box>
  );
};

export default DashboardCard;
