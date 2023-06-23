import { Slider } from '@mui/material';
import styled from 'styled-components';

export const ButtonContainer = styled.div`
  position: relative;
`;
export const InputContainer = styled.div`
  border: solid 1px rgba(0, 0, 0, 0.1);
  width: 200px;
  position: absolute;
  top: 0;
  left: 27px;
  height: 95%;
  padding: 0 6px 0;
`;

export const CustomSlider = styled(Slider)({
  height: 8,
  '& .MuiSlider-track': {
    border: 'none',
  },
  '& .MuiSlider-thumb': {
    height: 12,
    width: 12,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    '&:focus, &:hover, &.Mui-active, &.Mui-focusVisible': {
      boxShadow: 'inherit',
    },
    '&:before': {
      display: 'none',
    },
  },
  '& .MuiSlider-valueLabel': {
    lineHeight: 1.2,
    fontSize: 12,
    background: 'unset',
    padding: 0,
    width: 32,
    height: 32,
    borderRadius: '50% 50% 50% 0',
    backgroundColor: '#384E77',
    transformOrigin: 'bottom left',
    transform: 'translate(50%, -100%) rotate(-45deg) scale(0)',
    '&:before': { display: 'none' },
    '&.MuiSlider-valueLabelOpen': {
      transform: 'translate(50%, -100%) rotate(-45deg) scale(1)',
    },
    '& > *': {
      transform: 'rotate(45deg)',
    },
  },
});
