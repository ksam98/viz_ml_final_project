import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import './App.css';

const ErrorEvolutionPage = () => {
    const { errorType } = useParams();
    const navigate = useNavigate();

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [evolutionData, setEvolutionData] = useState([]);

    const epochs = [1, 5, 10]; // Your available epochs

    useEffect(() => {
        setLoading(true);
        
        // Fetch all epochs
        const promises = epochs.map(epoch =>
            fetch(`/data/results_epoch_${epoch}.json`)
                .then(res => {
                    if (!res.ok) throw new Error(`Failed to load Epoch ${epoch}`);
                    return res.json();
                })
                .then(data => processErrorData(data, errorType, epoch))
        );

        Promise.all(promises)
            .then(results => {
                // Combine and deduplicate images across epochs
                const allImages = new Map();
                
                results.forEach(epochData => {
                    epochData.images.forEach(img => {
                        if (!allImages.has(img.id)) {
                            allImages.set(img.id, {
                                id: img.id,
                                epochs: {}
                            });
                        }
                        allImages.get(img.id).epochs[epochData.epoch] = img.count;
                    });
                });

                // Convert to array and sort by total errors
                const processed = Array.from(allImages.values()).map(img => ({
                    ...img,
                    totalErrors: Object.values(img.epochs).reduce((sum, count) => sum + count, 0)
                })).sort((a, b) => b.totalErrors - a.totalErrors);

                setEvolutionData(processed);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setError(err.message);
                setLoading(false);
            });
    }, [errorType]);

    const processErrorData = (data, type, epoch) => {
        const typeMap = {
            'classification': 'Cls',
            'localization': 'Loc',
            'both': 'Both',
            'duplicate': 'Dupe',
            'background': 'Bkg',
            'miss': 'Miss'
        };

        const errorCode = typeMap[type.toLowerCase()];
        const imageCounts = {};

        data.mispredictions.forEach(item => {
            const count = item.errors.filter(e => e === errorCode).length;
            if (count > 0) {
                imageCounts[item.image_id] = count;
            }
        });

        return {
            epoch,
            images: Object.entries(imageCounts).map(([id, count]) => ({ id, count }))
        };
    };

    const resolveImagePath = (imageId) => {
        const id_len = imageId.toString().length;
        const padding = '0'.repeat(12 - id_len);
        const imageFileName = padding + imageId.toString();
        return '/images/' + imageFileName + '.jpg';
    };

    const handleImageClick = (imageId) => {
        navigate(`/image-evolution/${imageId}?errorType=${errorType}`);
    };

    if (loading) return <div className="loading-container"><div className="loading-spinner"></div></div>;
    if (error) return (
        <div className="error-container">
            <h2>Error</h2>
            <p>{error}</p>
            <button onClick={() => navigate('/')}>Back to Dashboard</button>
        </div>
    );

    return (
        <div className="app">
            <header className="app-header">
                <button className="back-button" onClick={() => navigate('/')}>← Back to Dashboard</button>
                <h1>{errorType.charAt(0).toUpperCase() + errorType.slice(1)} Error Evolution</h1>
                <p className="subtitle">Images with this error across all epochs (Epochs: {epochs.join(', ')})</p>
            </header>

            <div style={{ padding: '2rem' }}>
                <div style={{ 
                    marginBottom: '2rem', 
                    padding: '1rem', 
                    background: 'var(--surface)',
                    borderRadius: '8px',
                    border: '1px solid var(--border)'
                }}>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                        <strong>Total Images:</strong> {evolutionData.length}
                    </p>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                        Click any image to view its feature map evolution across epochs
                    </p>
                </div>

                <div className="image-grid">
                    {evolutionData.map(item => (
                        <div 
                            key={item.id} 
                            className="image-card" 
                            onClick={() => handleImageClick(item.id)}
                            style={{ cursor: 'pointer', position: 'relative' }}
                        >
                            <img 
                                src={resolveImagePath(item.id)} 
                                alt={`Image ${item.id}`} 
                            />
                            <div style={{
                                position: 'absolute',
                                bottom: '8px',
                                left: '8px',
                                right: '8px',
                                background: 'rgba(0,0,0,0.8)',
                                padding: '0.5rem',
                                borderRadius: '4px',
                                fontSize: '0.85rem'
                            }}>
                                <div style={{ color: 'white', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                                    Image {item.id}
                                </div>
                                <div style={{ color: '#cbd5e1', fontSize: '0.8rem' }}>
                                    {Object.entries(item.epochs).map(([ep, count]) => (
                                        <span key={ep} style={{ marginRight: '0.5rem' }}>
                                            E{ep}: {count}
                                        </span>
                                    ))}
                                </div>
                                
                                </div>
                            </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default ErrorEvolutionPage;