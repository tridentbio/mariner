import styled from 'styled-components';

export const EdgeButtonContainer = styled.div`
  background: transparent;
  width: 60px;
  height: 60px;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 60px;
  margin: 0;
`;

export const EdgeButton = styled.button`
  width: 30px;
  height: 30px;
  background: #e5e5e5;
  border: 1px solid #fff;
  cursor: pointer;
  border-radius: 50%;
  font-size: 16px;
  line-height: 1;
  color: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 0;
  &:hover {
    box-shadow: 0 0 6px 2px rgba(0, 0, 0, 0.08);
  }
`;
