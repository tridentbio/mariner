import { InfoRounded } from '@mui/icons-material';
import {
  Box,
  Popover,
  SvgIconProps,
  Tooltip,
  Link as MuiLink,
} from '@mui/material';
import IconButton from 'components/atoms/IconButton';
import AppLink from 'components/atoms/AppLink';
import { useEffect, useState } from 'react';
import { theme } from 'theme';
import CancelIcon from '@mui/icons-material/Cancel';
import HTMLMath from './HTMLMath';

/*  Using a CSS file here because it was the only way to add some custom style to MUI Popover component, i tried to aplly the style directly through props and also using Styled Components but for some reason it doesn't work */
import './paper.css';

type DocsModelProps = {
  docs?: string;
  docsLink?: string;
  commonIconProps?: SvgIconProps;
};

const DocsModel: React.FC<DocsModelProps> = ({
  docs,
  docsLink,
  commonIconProps,
}) => {
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);

  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };
  useEffect(() => {
    // @ts-ignore
    window.MathJax?.Hub?.Typeset();
  }, [anchorEl]);

  const open = Boolean(anchorEl);
  const id = open ? 'simple-popover' : undefined;
  return (
    <>
      <Tooltip title={'Documentation'}>
        <IconButton onClick={handleClick} className="nodrag">
          <InfoRounded
            sx={{ color: theme.palette?.primary.light }}
            {...commonIconProps}
          />
        </IconButton>
      </Tooltip>
      <Popover
        className="Popover"
        id={id}
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
      >
        <Box
          sx={{
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            position: 'relative',
          }}
        >
          <HTMLMath html={docs || ''} />
          {docsLink && (
            <MuiLink
              target="_blank"
              href={docsLink || '#'}
              sx={{ color: 'white' }}
            >
              Read More
            </MuiLink>
          )}
          <IconButton
            size="large"
            onClick={handleClose}
            sx={{ position: 'absolute', top: 10, right: 10 }}
          >
            <CancelIcon
              fontSize="large"
              sx={{ color: 'white', width: '25px', height: '25px' }}
              {...commonIconProps}
            />
          </IconButton>
        </Box>
      </Popover>
    </>
  );
};

export default DocsModel;
