import { NotificationContextProvider } from "@app/notifications";
import { store } from "@app/store";
import Notifications from "@components/organisms/Notifications";
import { ThemeProvider } from "@mui/material";
import { ReactNode } from "react";
import { Provider } from "react-redux";
import { MemoryRouter, MemoryRouterProps, Route, Routes } from "react-router-dom";
import { theme } from "theme";
export interface DefaultProvidersProps {
  children: ReactNode;
  routerProps?: MemoryRouterProps;
  routePath?: string;
}

export const DefaultProviders = (props: DefaultProvidersProps) => {
  const Base = () => (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <NotificationContextProvider>
          <Notifications />
          {props.children}
        </NotificationContextProvider>
      </ThemeProvider>
    </Provider>
  )

  return (
    <MemoryRouter {...props.routerProps}>
      <Routes>
        <Route element={<Base />} path={props.routePath ?? '/'} />
      </Routes>
    </MemoryRouter>
  )
}