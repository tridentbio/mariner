import styled from 'styled-components';
export const StyleOverrides = styled.div`
  .model_editor_error {
    stroke: rgba(255, 0, 0, 0.2);
  }
  .react-flow__node {
    border: 1px solid #cccccc;
    background: white;
    min-width: 250px;
    padding: 3px;
    display: flex;
    flex-direction: column;
    text-align: center;
    width: 250px;
    box-sizing: border-box;
    border-radius: 3px;

    .header {
      position: absolute;
      right: 0;
      top: 0;
      display: flex;
      flex-direction: row;
      justify-content: flex-end;
    }
    .content {
      margin-top: 10px;
      margin-bottom: 10px;
    }
  }
  .react-flow__node-layer,
  react-flow__node-featurizer {
  }
`;

export const NODE_DEFAULT_STYLISH: {
  borderColor: string;
} = {
  borderColor: 'rgb(204, 204, 204)',
};
