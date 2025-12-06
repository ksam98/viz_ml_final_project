import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './App.css';
import ErrorBarChart from './components/ErrorBarChart';
import ErrorPieChart from './components/ErrorPieChart';
import MetricsCard from './components/MetricsCard';

function Dashboard() {
    const navigate = useNavigate();
    const [selectedEpoch, setSelectedEpoch] = useState(1);
    const [allEpochsData, setAllEpochsData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        setLoading(true);

        // Fetch all available epochs (currently 1 and 2)
        const epochs = [1, 2];
        const promises = epochs.map(epoch =>
            fetch(`/data/results_epoch_${epoch}.json`)
                .then(res => {
                    if (!res.ok) throw new Error(`Failed to load Epoch ${epoch}`);
                    return res.json();
                })
                .then(data => processData(data, epoch))
        );

        Promise.all(promises)
            .then(results => {
                setAllEpochsData(results);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setError(err.message);
                setLoading(false);
            });
    }, []);

    const processData = (data, epoch) => {
        // Map raw JSON to the structure expected by components

        // 1. Metrics
        const metrics = {
            ap_50: data.metrics.mAP50,
            mAP: data.metrics.mAP,
        };

        // 2. Main Errors
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
            epoch: epoch,
            metadata: {
                model: "Faster R-CNN",
                dataset: "COCO Val 2017 (Subset)",
                num_images: 6,
                num_classes: 80
            },
            overall_metrics: metrics,
            main_errors: main_errors,
            raw_data: data // Keep raw data for drill-down
        };
    };

    const handleErrorClick = (errorType) => {
        // Navigate to error detail page
        // errorType comes from the chart key, e.g., "classification"
        navigate(`/error/${errorType}?epoch=${selectedEpoch}`);
    };

    // Derived state for the currently selected epoch
    const currentEpochData = allEpochsData.find(d => d.epoch === selectedEpoch);

    if (loading) {
        return (
            <div className="loading-container">
                <div className="loading-spinner"></div>
                <p>Loading Analysis...</p>
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

    if (!currentEpochData) return null;

    return (
        <div className="app">
            <header className="app-header">
                <div className="header-content">
                    <div>
                        <h1>TIDE Error Analysis Dashboard</h1>
                        <p className="subtitle">
                            {currentEpochData.metadata.model} on {currentEpochData.metadata.dataset}
                        </p>
                    </div>
                    <div className="epoch-selector">
                        <label htmlFor="epoch-select">Select Epoch: </label>
                        <select
                            id="epoch-select"
                            value={selectedEpoch}
                            onChange={(e) => setSelectedEpoch(Number(e.target.value))}
                        >
                            {allEpochsData.map(d => (
                                <option key={d.epoch} value={d.epoch}>Epoch {d.epoch}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </header>

            <div className="top-section">
                <div className="metrics-container">
                    <div className="metrics-grid">
                        <MetricsCard
                            title="mAP @ IoU=0.50"
                            value={(currentEpochData.overall_metrics.ap_50 * 100).toFixed(2)}
                            unit="%"
                            description="Average Precision at 50% IoU threshold"
                        />
                        <MetricsCard
                            title="mAP (COCO)"
                            value={(currentEpochData.overall_metrics.mAP * 100).toFixed(2)}
                            unit="%"
                            description="Mean Average Precision (IoU=.50:.05:.95)"
                        />
                        <MetricsCard
                            title="Images Evaluated"
                            value={currentEpochData.metadata.num_images}
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
                            <span className="error-val">{currentEpochData.main_errors.classification.toFixed(2)}</span>
                        </div>
                        <div className="error-summary-item">
                            <span className="error-label error-loc">Loc</span>
                            <span className="error-desc">Localization</span>
                            <span className="error-val">{currentEpochData.main_errors.localization.toFixed(2)}</span>
                        </div>
                        <div className="error-summary-item">
                            <span className="error-label error-both">Both</span>
                            <span className="error-desc">Cls + Loc</span>
                            <span className="error-val">{currentEpochData.main_errors.both.toFixed(2)}</span>
                        </div>
                        <div className="error-summary-item">
                            <span className="error-label error-dupe">Dupe</span>
                            <span className="error-desc">Duplicate</span>
                            <span className="error-val">{currentEpochData.main_errors.duplicate.toFixed(2)}</span>
                        </div>
                        <div className="error-summary-item">
                            <span className="error-label error-bkg">Bkg</span>
                            <span className="error-desc">Background</span>
                            <span className="error-val">{currentEpochData.main_errors.background.toFixed(2)}</span>
                        </div>
                        <div className="error-summary-item">
                            <span className="error-label error-miss">Miss</span>
                            <span className="error-desc">Missed</span>
                            <span className="error-val">{currentEpochData.main_errors.miss.toFixed(2)}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="visualizations-grid">
                <div className="viz-card">
                    <h2>Main Error Breakdown (Evolution)</h2>
                    <p className="viz-description">
                        Evolution of error impact (dAP) across epochs. Click an error type to drill down.
                    </p>
                    <ErrorBarChart
                        data={allEpochsData}
                        selectedEpoch={selectedEpoch}
                        onEpochSelect={setSelectedEpoch}
                        onErrorSelect={handleErrorClick}
                    />
                </div>

                <div className="viz-card">
                    <h2>Error Distribution (Epoch {selectedEpoch})</h2>
                    <p className="viz-description">
                        Relative proportion of each error type
                    </p>
                    <ErrorPieChart data={currentEpochData.main_errors} />
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

export default Dashboard;
