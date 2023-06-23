import { createTheme, ThemeOptions } from '@mui/material';

declare module '@mui/material/styles/createPalette' {
  interface PaletteOptions {
    tertiary: PaletteColorOptions;
  }
}

export const themeOptions: ThemeOptions = {
  palette: {
    primary: {
      main: '#384E77',
      dark: '#17294b',
    },
    secondary: {
      main: '#8BBEB2',
    },
    tertiary: {
      main: '#18314F',
      light: '#E6F9AF',
      dark: '#0D0630',
    },
  },
  typography: {
    fontSize: 18,
  },
};

export const theme = createTheme(themeOptions);
