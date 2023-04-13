import { Button } from '@mui/material';
import { FullScreen, useFullScreenHandle } from 'react-full-screen';
import styled from 'styled-components';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import { ReactNode, useEffect, useState } from 'react';

const FullScreenContainer = styled.div`
  .full-screen-button {
    position: absolute;
    top: ${({ fullScreen }: { fullScreen: boolean }) =>
      fullScreen ? 'unset' : '83vh'};
    right: 0;
    bottom: ${({ fullScreen }: { fullScreen: boolean }) =>
      fullScreen ? '1rem' : 'unset'};
    right: 0;
    z-index: 100;
  }
  .fullscreen-enabled {
    background: white;
  }
`;

type FullScreenWapperProps = {
  children: ReactNode;
  fullScreen: boolean;
  setFullScreen: (value: boolean) => void;
};

const FullScreenWrapper = ({
  children,
  fullScreen,
  setFullScreen,
}: FullScreenWapperProps) => {
  const fullScreenHandle = useFullScreenHandle();

  useEffect(() => {
    if (fullScreen) {
      fullScreenHandle.enter();
    } else {
      fullScreenHandle.exit();
    }
  }, [fullScreen]);

  return (
    <FullScreenContainer fullScreen={fullScreen}>
      <FullScreen handle={fullScreenHandle}>
        <>
          <Button
            onClick={() => setFullScreen(!fullScreen)}
            className="full-screen-button"
          >
            {fullScreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
          </Button>
          {children}
        </>
      </FullScreen>
    </FullScreenContainer>
  );
};

export default FullScreenWrapper;
