import styled from 'styled-components';

export const SeqContainer = styled.div`
  width: 80vw;
  height: 30vh;
  display: flex;
  flex-direction: row;
  gap: 2rem;
  .public-DraftEditor-content {
    width: 36.2vw;
    height: 14vh;
  }
  .la-vz-linear-scroller {
    overflow-y: auto;
  }
  .yellow {
    background-color: yellow;
  }
  .red {
    background-color: red;
  }
  .public-DraftStyleDefault-block {
    span {
      span {
        font-size: 1.5rem;
      }
    }
  }
`;

export const generateSeqvizSx = (visualization: 'linear' | 'circular') => ({
  width: '55vw',
  ...(visualization === 'circular'
    ? {
        height: '35vh',
        alignSelf: 'center',
        justifySelf: 'center',
      }
    : {}),
});

export const inputContainerSx = {
  border: '1px solid grey',
  borderRadius: '5px',
  padding: '1rem',
  margin: '1rem',
  marginBottom: '3rem',
  whiteSpace: 'nowrap',
  overflow: 'auto',
};
