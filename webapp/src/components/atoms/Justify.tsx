import styled from 'styled-components';

type IJustify = { position?: 'start' | 'end' | 'center' };

const Justify = styled.div`
  display: flex;
  justify-content: ${({ position }: IJustify) => position || 'unset'};
  padding-right: ${({ position }: IJustify) =>
    position === 'end' ? '2rem' : 'unset'};
`;
export default Justify;
