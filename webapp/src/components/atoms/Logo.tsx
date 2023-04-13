import { ImgHTMLAttributes } from 'react';
import { Box, BoxProps } from '@mui/system';

interface LogoProps extends BoxProps {
  imageProps?: Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'>;
}
const Logo = ({ sx, imageProps }: LogoProps) => (
  <Box sx={sx}>
    <img src="/mariner_logo.svg" width={100} {...imageProps} />
  </Box>
);

export default Logo;
