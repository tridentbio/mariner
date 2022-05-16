import React from 'react';
import {Routes, Route} from 'react-router-dom'
import { Counter } from './features/counter/Counter';
import './App.css';
import AuthenticationPage from './features/users/Authentication';

function App() {
  return (
    <div className="App">
      <Routes> 
        <Route path="/login" element={<AuthenticationPage/>}/>
      </Routes>
</div>
  );
}

export default App;
