import { useState, useEffect } from 'react';
import './App.css';
import ErrorBarChart from './components/ErrorBarChart';
import ErrorPieChart from './components/ErrorPieChart';
import MetricsCard from './components/MetricsCard';

function App() {
  const [tideData, setTideData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Load TIDE data from results folder
    fetch('/results/tide_results/tide_results.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to load TIDE data');
        }
        return response.json();
      })
      .then(data => {
        setTideData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading TIDE Analysis...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error Loading Data</h2>
        <p>{error}</p>
        <p className="error-hint">
          Make sure you've run the evaluation and TIDE analysis scripts first:
          <br />
          1. <code>python scripts/evaluate_fasterrcnn_voc.py</code>
          <br />
          2. <code>python scripts/tide_analysis_fasterrcnn.py --public_results_dir ./dashboard/public/results/tide_results</code>
        </p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>TIDE Error Analysis Dashboard</h1>
        <p className="subtitle">
          {tideData.metadata.model} on {tideData.metadata.dataset}
        </p>
      </header>

      <div className="metrics-grid">
        <MetricsCard
          title="AP @ IoU=0.50"
          value={(tideData.overall_metrics.ap_50 * 100).toFixed(2)}
          unit="%"
          description="Average Precision at 50% IoU threshold"
        />
        <MetricsCard
          title="Images Evaluated"
          value={tideData.metadata.num_images}
          description="Total images in validation set"
        />
        <MetricsCard
          title="Classes"
          value={tideData.metadata.num_classes}
          description="Number of object categories"
        />
      </div>

      <div className="visualizations-grid">
        <div className="viz-card">
          <h2>Main Error Breakdown</h2>
          <p className="viz-description">
            Impact of each error type on Average Precision (dAP)
          </p>
          <ErrorBarChart data={tideData.main_errors} />
        </div>

        <div className="viz-card">
          <h2>Error Distribution</h2>
          <p className="viz-description">
            Relative proportion of each error type
          </p>
          <ErrorPieChart data={tideData.main_errors} />
        </div>
      </div>

      <div className="error-details">
        <h2>Error Type Descriptions</h2>
        <div className="error-grid">
          <div className="error-item">
            <h3 className="error-cls">Classification Error</h3>
            <p>Correct localization but wrong class prediction</p>
            <span className="error-value">
              dAP: {(tideData.main_errors.classification * 100).toFixed(2)}%
            </span>
          </div>
          <div className="error-item">
            <h3 className="error-loc">Localization Error</h3>
            <p>Correct class but poor bounding box overlap</p>
            <span className="error-value">
              dAP: {(tideData.main_errors.localization * 100).toFixed(2)}%
            </span>
          </div>
          <div className="error-item">
            <h3 className="error-both">Both Errors</h3>
            <p>Wrong class AND poor localization</p>
            <span className="error-value">
              dAP: {(tideData.main_errors.both * 100).toFixed(2)}%
            </span>
          </div>
          <div className="error-item">
            <h3 className="error-dupe">Duplicate Detection</h3>
            <p>Multiple detections on the same object</p>
            <span className="error-value">
              dAP: {(tideData.main_errors.duplicate * 100).toFixed(2)}%
            </span>
          </div>
          <div className="error-item">
            <h3 className="error-bkg">Background Error</h3>
            <p>False positive on background regions</p>
            <span className="error-value">
              dAP: {(tideData.main_errors.background * 100).toFixed(2)}%
            </span>
          </div>
          <div className="error-item">
            <h3 className="error-miss">Missed Detection</h3>
            <p>Ground truth object not detected</p>
            <span className="error-value">
              dAP: {(tideData.main_errors.miss * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      <footer className="app-footer">
        <p>
          TIDE: A General Toolbox for Identifying Object Detection Errors
          <br />
          <a href="https://dbolya.github.io/tide/" target="_blank" rel="noopener noreferrer">
            Learn more about TIDE
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
