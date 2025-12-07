import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './Dashboard';
import ErrorDetailPage from './ErrorDetailPage';
import ImageDetail from './ImageDetail';

import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/error/:errorType" element={<ErrorDetailPage />} />
        <Route path="/image/:imageId" element={<ImageDetail />} />
        </Routes>
    </Router>
  );
}

export default App;
