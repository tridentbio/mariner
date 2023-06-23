import { ThemeProvider } from '@emotion/react';
import { Meta } from '@storybook/react';
import { useState } from 'react';
import { BrowserRouter } from 'react-router-dom';
import styled from 'styled-components';
import { theme } from 'theme';
import { BiologicalInput } from '.';

const Container = styled.div`
  width: 100%;
  height: 100%;
`;

const storyMeta: Meta = {
  title: 'Components Biological Input',
  component: BiologicalInput,
  decorators: [
    (Story) => {
      return (
        <ThemeProvider theme={theme}>
          <BrowserRouter>
            <Container>{Story()}</Container>
          </BrowserRouter>
        </ThemeProvider>
      );
    },
  ],
};
export default storyMeta;

const BioInput = ({
  domainKind,
}: {
  domainKind: 'dna' | 'rna' | 'protein';
}) => {
  const [value, setValue] = useState<string>('');

  return (
    <BiologicalInput
      value={value}
      onChange={setValue}
      domainKind={domainKind}
      key=""
      label=""
    />
  );
};

export const DNA = () => <BioInput domainKind="dna" />;

export const RNA = () => <BioInput domainKind="rna" />;

export const Protein = () => <BioInput domainKind="protein" />;
