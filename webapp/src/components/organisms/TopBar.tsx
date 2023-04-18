import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  ListItem,
  ListItemText,
  IconButton,
  Drawer,
  Divider,
  List,
  ListItemButton,
  Link as MuiLink,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import styled from 'styled-components';
import { logout } from '../../features/users/usersSlice';
import { useAppDispatch } from '../../app/hooks';
import Logo from 'components/atoms/Logo';
import AppLink from 'components/atoms/AppLink';

const DrawerHeader = styled.div`
  display: flex;
  align-items: center;
  padding: 1;
  justify-content: flex-end;
`;

const Bottom = styled.div`
  margin: auto 0 0 0;
`;

const menuItems = [
  {
    title: 'Dashboard',
    link: '/',
  },
  {
    title: 'Datasets',
    link: '/datasets',
  },
  {
    title: 'Models',
    link: '/models',
  },
  {
    title: 'Trainings',
    link: '/trainings',
  },
];

export const TopBar = () => {
  const dispatch = useAppDispatch();

  const navigate = useNavigate();

  const [openedDrawer, setOpenedDrawer] = useState(false);

  const handleDrawerOpen = () => setOpenedDrawer(true);

  const handleDrawerClose = () => setOpenedDrawer(false);

  const drawerWidth = 400;

  const handleLogout: React.MouseEventHandler<HTMLLIElement> = () => {
    dispatch(logout());
    navigate('/login');
  };

  return (
    <div>
      <AppBar sx={{ backgroundColor: 'primary.main' }} position="static">
        <Toolbar>
          <IconButton
            aria-label="open drawer"
            onClick={handleDrawerOpen}
            edge="start"
            sx={{
              mr: 2,
              ...(openedDrawer && { display: 'none' }),
            }}
          >
            <MenuIcon sx={{ color: 'white' }} />
          </IconButton>
          <AppLink to="/">
            <Logo />
          </AppLink>
        </Toolbar>
      </AppBar>
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
          backgroundColor: 'primary',
        }}
        anchor="left"
        open={openedDrawer}
        onClose={() => setOpenedDrawer(false)}
      >
        <DrawerHeader>
          <IconButton onClick={handleDrawerClose}>
            <ChevronLeftIcon />
          </IconButton>
        </DrawerHeader>
        <Divider />
        <List>
          {menuItems.map(({ link, title }) => (
            <MuiLink
              component={Link}
              to={link}
              onClick={() => setOpenedDrawer(false)}
              key={title}
              style={{ textDecoration: 'none' }}
            >
              <ListItem disablePadding>
                <ListItemButton>
                  <ListItemText primary={title} />
                </ListItemButton>
              </ListItem>
            </MuiLink>
          ))}
        </List>
        <Divider />
        <Bottom>
          <List>
            <ListItem key={'Log out'} disablePadding onClick={handleLogout}>
              <ListItemButton>
                <ListItemText primary={'Log out'} />
              </ListItemButton>
            </ListItem>
          </List>
        </Bottom>
      </Drawer>
    </div>
  );
};
