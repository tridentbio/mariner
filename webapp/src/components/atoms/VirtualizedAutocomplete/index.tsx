import Autocomplete, { AutocompleteProps, autocompleteClasses } from '@mui/material/Autocomplete';
import Popper from '@mui/material/Popper';
import { styled } from '@mui/material/styles';
import { CSSProperties, Fragment, HTMLAttributes, ReactNode, Ref, createContext, forwardRef, useContext, useEffect, useRef } from 'react';
import { VariableSizeList } from 'react-window';

const StyledPopper = styled(Popper)({
  [`& .${autocompleteClasses.listbox}`]: {
    boxSizing: 'border-box',
    '& ul': {
      padding: 0,
      margin: 0,
    },
  },
});

/* eslint-disable react/prop-types */
const Row = ({ data, index, style }: {data: ReactNode[], index: number, style: CSSProperties}) => {
  const elem = data[index];
  return (
    <div style={style}>
      <Fragment>{elem}</Fragment>
    </div>
  );
};

const useResetCache = (data: any) => {
  const ref = useRef<VariableSizeList>(null);
  
  useEffect(() => {
    if (ref.current !== null) {
      ref.current.resetAfterIndex(0, true);
    }
  }, [data]);

  return ref;
};

const OuterElementContext = createContext({});

// eslint-disable-next-line react/display-name
const OuterElementType = forwardRef<HTMLDivElement>((props, ref) => {
  const outerProps = useContext(OuterElementContext);
  return <div ref={ref} {...props} {...outerProps} />;
});

type VirtualizedListProps = {
  rowheight: number;
  htmlProps: HTMLAttributes<HTMLElement>;
};

// eslint-disable-next-line react/display-name
export const VirtualizedList = forwardRef((props: VirtualizedListProps, ref: Ref<HTMLDivElement>) => {
  const itemCount = (props.htmlProps.children as ReactNode[]).length;
  const gridRef = useResetCache(itemCount);
  const outerProps = { ...props.htmlProps };

  delete outerProps.children;
  
  return (
    <div ref={ref}>
      <OuterElementContext.Provider value={outerProps}>
        <VariableSizeList
          ref={gridRef}
          outerElementType={OuterElementType}
          className="List"
          width={'100%'}
          height={400}
          itemCount={itemCount}
          itemSize={() => props.rowheight}
          overscanCount={5}
          itemData={{ ...(props.htmlProps.children as ReactNode[]) }}
        >
          {Row}
        </VariableSizeList>
      </OuterElementContext.Provider>
    </div>
  );
});


interface VirtualizedAutoCompleteProps<
  T,
  Multiple extends boolean | undefined = false,
  DisableClearable extends boolean | undefined = false,
  FreeSolo extends boolean | undefined = false,
> extends Omit<AutocompleteProps<T, Multiple, DisableClearable, FreeSolo>, 'ListBoxComponent'> {
  boxPadding?: number;
}

// eslint-disable-next-line react/display-name
const VirtualizedListComp = forwardRef((listProps: HTMLAttributes<HTMLElement>, ref: Ref<HTMLDivElement>) => {
  return <VirtualizedList ref={ref} htmlProps={listProps} rowheight={48} />;
});

export const VirtualizedAutocomplete = <
  T,
  Multiple extends boolean | undefined = false,
  DisableClearable extends boolean | undefined = false,
  FreeSolo extends boolean | undefined = false,
>(props: VirtualizedAutoCompleteProps<T, Multiple, DisableClearable, FreeSolo>) => {
  return (
    <Autocomplete
      {...props}
      disableListWrap
      PopperComponent={StyledPopper}
      ListboxComponent={VirtualizedListComp}
      // TODO: Post React 18 update - validate this conversion, look like a hidden bug
    />
  );
}