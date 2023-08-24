import type { TestingLibraryMatchers } from '@testing-library/jest-dom/matchers';

declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {}
}
declare global {
  namespace jest {
    interface Matchers<R = void>
      extends TestingLibraryMatchers<typeof expect.stringContaining, R> {}
  }
}
