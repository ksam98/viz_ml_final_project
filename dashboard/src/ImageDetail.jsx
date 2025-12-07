import { useState, useEffect } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import './App.css';

function ImageDetail() {
    const { imageId } = useParams();
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const epoch = searchParams.get('epoch') || '1';

    const [images, setImages] = useState({ backbone: [], fpn: [], pool: [] });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fpnLayers = 3;

    useEffect(() => {
        setLoading(true);
        setError(null);

        try {
            const loadedImages = { backbone: [], fpn: [], pool: [] };

            // Backbone Grad-CAM (single image)
            loadedImages.backbone = [`/data/gradcams/epoch_${epoch}/backbone/img_${imageId}.png`];

            // FPN maps (multiple layers)
            loadedImages.fpn = Array.from({ length: fpnLayers }, (_, i) => 
                `/data/gradcams/epoch_${epoch}/fpn/${i}_img_${imageId}.png`
            );

            // Pool image
            loadedImages.pool = [`/data/gradcams/epoch_${epoch}/fpn/pool_img_${imageId}.png`];

            console.log('Loaded image paths:', loadedImages);
            setImages(loadedImages);
            setLoading(false);
        } catch (err) {
            console.error(err);
            setError('Failed to load images.');
            setLoading(false);
        }
    }, [epoch, imageId]);

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
                <h1>Image Detail: Feature Map Flow</h1>
                <p className="subtitle">Epoch {epoch} • Image ID {imageId}</p>
            </header>

            <div className="image-detail-container">
                <div className="image-detail-grid">
                    {/* FPN Layers */}
                    {images.fpn.map((src, idx) => (
                        <div key={idx} className="image-column">
                            <h3>FPN Layer {idx}</h3>
                            <img 
                                src={src} 
                                alt={`FPN Layer ${idx}`}
                                onError={(e) => console.error(`Failed to load: ${src}`)}
                                onLoad={() => console.log(`Loaded: ${src}`)}
                            />
                        </div>
                    ))}

                    {/* Pool Layer */}
                    {images.pool.map((src, idx) => (
                        <div key={idx} className="image-column">
                            <h3>Pool Layer</h3>
                            <img 
                                src={src} 
                                alt="Pool Layer"
                                onError={(e) => console.error(`Failed to load: ${src}`)}
                                onLoad={() => console.log(`Loaded: ${src}`)}
                            />
                        </div>
                    ))}

                    {/* Backbone Grad-CAM */}
                    {images.backbone.map((src, idx) => (
                        <div key={idx} className="image-column">
                            <h3>Backbone Grad-CAM</h3>
                            <img 
                                src={src} 
                                alt={`Backbone Grad-CAM ${idx}`}
                                onError={(e) => console.error(`Failed to load: ${src}`)}
                                onLoad={() => console.log(`Loaded: ${src}`)}
                            />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default ImageDetail;