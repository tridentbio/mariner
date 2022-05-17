import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AppBar, Typography, Toolbar, ListItem, ListItemText, IconButton, Drawer, Divider, List, ListItemButton } from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft'
import styled from 'styled-components'
import { logout } from '../features/users/usersSlice'
import { useAppDispatch } from '../app/hooks'

const DrawerHeader = styled.div`
  display: flex;
  alignItems: center;
  padding: 1;
  justifyContent: flex-end;
`

const Bottom = styled.div`
  margin: auto 0 0 0;
`

const menuItems = [
  {
    title: 'Datasets',
    link: '/datasets'
  }
]

export const TopBar = () => {
  const dispatch = useAppDispatch()
  const navigate = useNavigate()
  const [openedDrawer, setOpenedDrawer] = useState(false)
  const handleDrawerOpen = () => setOpenedDrawer(true)
  const handleDrawerClose = () => setOpenedDrawer(false)
  const drawerWidth = 400
  const handleLogout: React.MouseEventHandler<HTMLLIElement> = () => {
    dispatch(logout())
    navigate('/login')
  }
  return (
    <div>
      <AppBar position="static">
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={handleDrawerOpen}
            edge="start"
            sx={{ mr: 2, ...(openedDrawer && { display: 'none' }) }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div">
            Mariner App
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box'
          }
        }}
        variant="persistent"
        anchor="left"
        open={openedDrawer}
      >
        <DrawerHeader>
          <IconButton onClick={handleDrawerClose}>
            <ChevronLeftIcon />
          </IconButton>
        </DrawerHeader>
        <Divider />
        <List>
          {menuItems.map(({ link, title }) => (
            <ListItem key={title} disablePadding>
              <Link to={link} style={{ textDecoration: 'none' }}>
                <ListItemButton>
                  <ListItemText primary={title} />
                </ListItemButton>
              </Link>
            </ListItem>
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
  )
}
