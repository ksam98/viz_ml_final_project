import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './Dashboard';
import ErrorDetailPage from './ErrorDetailPage';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/error/:errorType" element={<ErrorDetailPage />} />
      </Routes>
    </Router>
  );
}

export default App;
