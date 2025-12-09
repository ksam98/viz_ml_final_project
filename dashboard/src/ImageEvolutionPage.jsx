import { useState, useEffect } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import './App.css';

function ImageEvolutionPage() {
    const { imageId } = useParams();
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const errorType = searchParams.get('errorType') || 'unknown';

    const [allEpochImages, setAllEpochImages] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const epochs = [1, 5, 10];
    const fpnLayers = 3;

    useEffect(() => {
        setLoading(true);
        setError(null);

        try {
            const epochsData = epochs.map(epoch => {
                const epochImages = {
                    epoch,
                    backbone: `/data/gradcams/epoch_${epoch}/backbone/img_${imageId}.png`,
                    fpn: Array.from({ length: fpnLayers }, (_, i) => 
                        `/data/gradcams/epoch_${epoch}/fpn/${i}_img_${imageId}.png`
                    ),
                    pool: `/data/gradcams/epoch_${epoch}/fpn/pool_img_${imageId}.png`
                };
                return epochImages;
            });

            console.log('Loaded evolution paths:', epochsData);
            setAllEpochImages(epochsData);
            setLoading(false);
        } catch (err) {
            console.error(err);
            setError('Failed to load images.');
            setLoading(false);
        }
    }, [imageId]);

    if (loading) return <div className="loading-container"><div className="loading-spinner"></div></div>;
    if (error) return (
        <div className="error-container">
            <h2>{error}</h2>
            <button onClick={() => navigate(-1)}>Go Back</button>
        </div>
    );

    return (
        <div className="app">
            <header className="app-header">
                <button className="back-button" onClick={() => navigate(-1)}>← Back</button>
                <h1>Image Evolution: Feature Map Analysis</h1>
                <p className="subtitle">
                    Image ID {imageId} • Error Type: {errorType.charAt(0).toUpperCase() + errorType.slice(1)} • All Epochs
                </p>
            </header>

            <div style={{ padding: '2rem' }}>
                {allEpochImages.map(epochData => (
                    <div key={epochData.epoch} style={{ marginBottom: '4rem' }}>
                        <div style={{
                            background: 'var(--surface)',
                            padding: '1rem',
                            borderRadius: '8px',
                            marginBottom: '1.5rem',
                            border: '2px solid var(--border)'
                        }}>
                            <h2 style={{ 
                                color: 'var(--primary-color)', 
                                fontSize: '1.5rem',
                                marginBottom: '0.25rem'
                            }}>
                                Epoch {epochData.epoch}
                            </h2>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                Feature maps and attention visualizations
                            </p>
                        </div>

                        <div className="image-detail-container">
                            <div className="image-detail-grid">
                                {/* FPN Layers */}
                                {epochData.fpn.map((src, idx) => (
                                    <div key={`fpn-${idx}`} className="image-column">
                                        <h3>FPN Res {idx}</h3>
                                        <img 
                                            src={src} 
                                            alt={`Epoch ${epochData.epoch} FPN Layer ${idx}`}
                                            onError={(e) => {
                                                console.error(`Failed to load: ${src}`);
                                                e.target.style.display = 'none';
                                            }}
                                            onLoad={() => console.log(`Loaded: ${src}`)}
                                        />
                                    </div>
                                ))}

                                {/* Pool Layer */}
                                <div className="image-column">
                                    <h3>Pool Layer</h3>
                                    <img 
                                        src={epochData.pool} 
                                        alt={`Epoch ${epochData.epoch} Pool Layer`}
                                        onError={(e) => {
                                            console.error(`Failed to load: ${epochData.pool}`);
                                            e.target.style.display = 'none';
                                        }}
                                        onLoad={() => console.log(`Loaded: ${epochData.pool}`)}
                                    />
                                </div>

                                {/* Backbone Grad-CAM */}
                                <div className="image-column">
                                    <h3>Bacbone GradCAM</h3>
                                    <img 
                                        src={epochData.backbone} 
                                        alt={`Epoch ${epochData.epoch} Backbone Grad-CAM`}
                                        onError={(e) => {
                                            console.error(`Failed to load: ${epochData.backbone}`);
                                            e.target.style.display = 'none';
                                        }}
                                        onLoad={() => console.log(`Loaded: ${epochData.backbone}`)}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Divider between epochs */}
                        {epochData.epoch !== epochs[epochs.length - 1] && (
                            <div style={{
                                height: '2px',
                                background: 'linear-gradient(to right, transparent, var(--border), transparent)',
                                marginTop: '3rem'
                            }}></div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}

export default ImageEvolutionPage;