import { NotificationContextProvider } from "@app/notifications";
import { store } from "@app/store";
import Notifications from "@components/organisms/Notifications";
import { ThemeProvider } from "@mui/material";
import { PropsWithChildren } from "react";
import { Provider } from "react-redux";
import { BrowserRouter } from "react-router-dom";
import { theme } from "theme";

export const DefaultProviders = (props: PropsWithChildren) => (
  <BrowserRouter>
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <NotificationContextProvider>
          <Notifications />
          {props.children}
        </NotificationContextProvider>
      </ThemeProvider>
    </Provider>
  </BrowserRouter>
)