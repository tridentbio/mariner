import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '@app/store';

export { default as useAppNavigation } from './useAppNavigation'
export { usePopoverState } from './usePopoverState'
export { useToggle } from './useToogle'
export { default as useModelEditor } from './useModelEditor'


// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch = () => useDispatch<AppDispatch>();

export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
