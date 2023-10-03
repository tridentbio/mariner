import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import { Button } from '@mui/material';
import { ReactNode, createContext, useContext, useEffect } from 'react';
import {
  FullScreen,
  FullScreenHandle,
  useFullScreenHandle,
} from 'react-full-screen';
import styled from 'styled-components';

const FullScreenContainer = styled.div`
  position: relative;

  .full-screen-button {
    position: absolute;
    top: ${({ fullScreen }: { fullScreen: boolean }) =>
      fullScreen ? 'unset' : '75vh'};
    right: 0;
    bottom: ${({ fullScreen }: { fullScreen: boolean }) =>
      fullScreen ? '1rem' : 'unset'};
    right: 0;
    z-index: 100;
  }
  .fullscreen-enabled {
    background: white;
  }

  section {
    position: absolute;
    right: 50vw;
    bottom: 2rem;
  }
`;

type FullScreenWapperProps = {
  children: ReactNode;
  fullScreen: boolean;
  setFullScreen: (value: boolean) => void;
};

// @ts-ignore
const FullScreenContext = createContext<FullScreenHandle>({});

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
    <FullScreenContext.Provider value={fullScreenHandle}>
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
    </FullScreenContext.Provider>
  );
};

export default FullScreenWrapper;

export const useFullScreen = () => {
  const value = useContext(FullScreenContext);

  return value;
};
