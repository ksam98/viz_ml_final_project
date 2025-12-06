import { useState, useEffect } from 'react';
import './App.css';
import ErrorBarChart from './components/ErrorBarChart';
import ErrorPieChart from './components/ErrorPieChart';
import MetricsCard from './components/MetricsCard';

function App() {
  const [selectedEpoch, setSelectedEpoch] = useState(1);
  const [tideData, setTideData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    // Load TIDE data from results folder based on selected epoch
    fetch(`/data/results_epoch_${selectedEpoch}.json`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load data for Epoch ${selectedEpoch}`);
        }
        return response.json();
      })
      .then(data => {
        const processed = processData(data, selectedEpoch);
        setTideData(processed);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError(err.message);
        setLoading(false);
      });
  }, [selectedEpoch]);

  const processData = (data, epoch) => {
    // Map raw JSON to the structure expected by components

    // 1. Metrics
    const metrics = {
      ap_50: data.metrics.mAP50,
      mAP: data.metrics.mAP,
      // Add others if needed
    };

    // 2. Main Errors
    // The key in data.tide.errors.main depends on the epoch, e.g., "epoch1_predictions"
    const predictionKey = `epoch${epoch}_predictions`;
    const rawErrors = data.tide.errors.main[predictionKey] || {};

    const main_errors = {
      classification: rawErrors.Cls || 0,
      localization: rawErrors.Loc || 0,
      both: rawErrors.Both || 0,
      duplicate: rawErrors.Dupe || 0,
      background: rawErrors.Bkg || 0,
      miss: rawErrors.Miss || 0,
    };

    return {
      metadata: {
        model: "Faster R-CNN", // Hardcoded or extracted if available
        dataset: "COCO Val 2017 (Subset)",
        num_images: 6, // We know it's 6 images
        num_classes: 80 // COCO default
      },
      overall_metrics: metrics,
      main_errors: main_errors
    };
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading Analysis for Epoch {selectedEpoch}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error Loading Data</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div>
            <h1>TIDE Error Analysis Dashboard</h1>
            <p className="subtitle">
              {tideData.metadata.model} on {tideData.metadata.dataset}
            </p>
          </div>
          <div className="epoch-selector">
            <label htmlFor="epoch-select">Select Epoch: </label>
            <select
              id="epoch-select"
              value={selectedEpoch}
              onChange={(e) => setSelectedEpoch(Number(e.target.value))}
            >
              <option value={1}>Epoch 1</option>
              <option value={2}>Epoch 2</option>
            </select>
          </div>
        </div>
      </header>

      <div className="top-section">
        <div className="metrics-container">
          <div className="metrics-grid">
            <MetricsCard
              title="mAP @ IoU=0.50"
              value={(tideData.overall_metrics.ap_50 * 100).toFixed(2)}
              unit="%"
              description="Average Precision at 50% IoU threshold"
            />
            <MetricsCard
              title="mAP (COCO)"
              value={(tideData.overall_metrics.mAP * 100).toFixed(2)}
              unit="%"
              description="Mean Average Precision (IoU=.50:.05:.95)"
            />
            <MetricsCard
              title="Images Evaluated"
              value={tideData.metadata.num_images}
              description="Total images in validation set"
            />
          </div>
        </div>

        <div className="error-summary-container">
          <h3>Error Types & Impact (dAP)</h3>
          <div className="error-summary-list">
            <div className="error-summary-item">
              <span className="error-label error-cls">Cls</span>
              <span className="error-desc">Classification</span>
              <span className="error-val">{tideData.main_errors.classification.toFixed(2)}</span>
            </div>
            <div className="error-summary-item">
              <span className="error-label error-loc">Loc</span>
              <span className="error-desc">Localization</span>
              <span className="error-val">{tideData.main_errors.localization.toFixed(2)}</span>
            </div>
            <div className="error-summary-item">
              <span className="error-label error-both">Both</span>
              <span className="error-desc">Cls + Loc</span>
              <span className="error-val">{tideData.main_errors.both.toFixed(2)}</span>
            </div>
            <div className="error-summary-item">
              <span className="error-label error-dupe">Dupe</span>
              <span className="error-desc">Duplicate</span>
              <span className="error-val">{tideData.main_errors.duplicate.toFixed(2)}</span>
            </div>
            <div className="error-summary-item">
              <span className="error-label error-bkg">Bkg</span>
              <span className="error-desc">Background</span>
              <span className="error-val">{tideData.main_errors.background.toFixed(2)}</span>
            </div>
            <div className="error-summary-item">
              <span className="error-label error-miss">Miss</span>
              <span className="error-desc">Missed</span>
              <span className="error-val">{tideData.main_errors.miss.toFixed(2)}</span>
            </div>
          </div>
        </div>
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

