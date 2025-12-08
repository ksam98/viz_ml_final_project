import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useNavigate, useLocation } from 'react-router-dom';
import './App.css';

function ImageDetail() {
    const { imageId, errorType } = useParams();
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const location = useLocation();
    const passedErrorCount = location.state?.errorCount;

    const epoch = searchParams.get('epoch') || '1';
    const [hoveredBoxIndex, setHoveredBoxIndex] = useState(-1);
    const [images, setImages] = useState({ backbone: [], fpn: [], pool: [] });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [imageErrorData, setImageErrorData] = useState([]);
    const canvasRef = useRef(null);
    const imgRef = useRef(null);
    const fpnLayers = 3;
    const resolveImagePath = (imageId) => {
        const id_len = imageId.toString().length;
        const padding = '0'.repeat(12 - id_len);
        const imageFileName = padding + imageId.toString();
        return '/images/' + imageFileName + '.jpg';
    }

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

            const promises = [
                fetch(`/data/results_epoch_${epoch}.json`)
                    .then(res => {
                        if (!res.ok) throw new Error(`Failed to load data for Epoch ${epoch}`);
                        return res.json();
                    })
                    .then(data => processData(data, imageId, errorType)),

                fetch(`/data/val.json`)
                    .then(res => {
                        if (!res.ok) throw new Error('Failed to load val.json');
                        return res.json();
                    })
                    .then(data => processValData(data, imageId))
            ]
            Promise.all(promises)
                .then(results => {
                    setImageErrorData([...results[0], ...results[1]]);
                    console.log('Processed val data for ImageDetail:', [...results[0], ...results[1]]);
                })
                .catch(err => {
                    console.error(err);
                    setError('Failed to load image data.');
                });
        } catch (err) {
            console.error(err);
            setError('Failed to load images.');
            setLoading(false);
        }
    }, [epoch, imageId]);
    const processValData = (data, imageId) => {
        let imageErrorMetadata = []
        for (let i = 0; i < data.annotations.length; i++) {
            if (data.annotations[i].image_id.toString() === imageId.toString()) {
                imageErrorMetadata.push(
                    {
                        confidence: 1,
                        predicted_classes: data.annotations[i].category_id,
                        box: data.annotations[i].bbox,
                        color: '#08f808'
                    }
                );
            }
        }

        return imageErrorMetadata;
    }
    const processData = (data, imageId, errorType) => {
        const typeMap = {
            'classification': 'Cls',
            'localization': 'Loc',
            'both': 'Both',
            'duplicate': 'Dupe',
            'background': 'Bkg',
            'miss': 'Miss'
        };

        const errorCode = typeMap[errorType.toLowerCase()];
        if (!errorCode) {
            throw new Error(`Unknown error type: ${errorType}`);
        }
        if (errorCode === "Bkg") {
            return [];
        }

        const imageData = data.mispredictions.find(item => item.image_id.toString() === imageId.toString());
        if (!imageData) {
            throw new Error(`Image ID ${imageId} not found in data.`);
        }
        let index = [];
        let imageErrorMetadata = [];
        for (let i = 0; i < imageData.errors.length; i++) {
            if (imageData.errors[i] === errorCode) {
                index.push(i);
                imageErrorMetadata.push(
                    {
                        confidence: imageData.confidences[i],
                        predicted_classes: imageData.predicted_classes[i],
                        box: imageData.boxes[i],
                        color: '#ee0a0a'
                    }
                );
            }
            if (i >= 5) break;  // Safety break to avoid too many boxes
        }

        return imageErrorMetadata
    };

    // Helper function to convert hex color to rgba format with specified opacity
    const hexToRgbA = (hex, opacity) => {
        let c;
        if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
            c = hex.substring(1).split('');
            if (c.length === 3) {
                c = [c[0], c[0], c[1], c[1], c[2], c[2]];
            }
            c = '0x' + c.join('');
            return 'rgba(' + [(c >> 16) & 255, (c >> 8) & 255, c & 255].join(',') + ',' + opacity + ')';
        }
        // Fallback if the input isn't a valid hex
        return `rgba(0, 0, 0, ${opacity})`;
    };

    const drawBoundingBoxes = () => {
        if (!imgRef.current || !canvasRef.current || imageErrorData.length === 0) return;

        const canvas = canvasRef.current;
        const img = imgRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas dimensions to match image
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw each bounding box
        imageErrorData.forEach((errorData, index) => {
            const box = errorData.box;
            const confidence = errorData.confidence;
            const predictedClass = errorData.predicted_classes;
            const className = `Class ${predictedClass}`;

            // Extract box coordinates [x1, y1, x2, y2]
            const [x1, y1, x2, y2] = box;
            const width = x2 - x1;
            const height = y2 - y1;

            // Determine opacity: full opacity for hovered box, 0.4 for others, 1 if no box is hovered.
            let opacity = 1;
            if (hoveredBoxIndex !== -1) {
                opacity = (hoveredBoxIndex === index) ? 1.0 : 0.1;
            }
            const rgbaColor = hexToRgbA(errorData.color, opacity);

            // Draw rectangle
            ctx.strokeStyle = rgbaColor;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, width, height);

            // Draw label background
            const label = `${className}: ${(confidence * 100).toFixed(1)}%`;
            ctx.font = 'bold 14px Arial';
            ctx.fillStyle = rgbaColor;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x1, y1 - 25, textWidth + 8, 24);

            // Draw label text (always full opacity white text for readability)
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x1 + 4, y1 - 8);
        });
    };

    const handleMouseMove = (event) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        // Calculate mouse position relative to canvas coordinates
        const mouseX = (event.clientX - rect.left) * scaleX;
        const mouseY = (event.clientY - rect.top) * scaleY;

        let newHoverIndex = -1;

        for (let i = 0; i < imageErrorData.length; i++) {
            const box = imageErrorData[i].box;
            const [x1, y1, x2, y2] = box;
            if (mouseX >= x2 + 5 && mouseX <= x1 - 5 && mouseY >= y2 + 5 && mouseY <= y1 - 5) {
                newHoverIndex = i;
                break;
            }
        }

        if (newHoverIndex !== hoveredBoxIndex) {
            setHoveredBoxIndex(newHoverIndex);
        }
    };

    // Use effect to draw when image loads AND when hover state changes
    useEffect(() => {
        if (imgRef.current && imgRef.current.complete) {
            drawBoundingBoxes();
        }
    }, [imageErrorData, hoveredBoxIndex]); // Added hoveredBoxIndex as dependency

    // Add event listeners when component mounts or dependencies change
    useEffect(() => {
        const canvas = canvasRef.current;
        if (canvas) {
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseleave', () => setHoveredBoxIndex(-1));
        }

        return () => {
            if (canvas) {
                canvas.removeEventListener('mousemove', handleMouseMove);
                // remove the mouseleave listener
                // canvas.removeEventListener('mouseleave', () => setHoveredBoxIndex(-1));
            }
        };
    }, [imageErrorData, hoveredBoxIndex]); // Dependencies for listeners


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
                <div>
                    <button
                        className="back-button"
                        onClick={() => navigate(-1)}
                        style={{
                            padding: '0.4rem 0.8rem',
                            fontSize: '0.9rem',
                            marginBottom: '1rem',
                            cursor: 'pointer',
                            background: 'var(--surface)',
                            color: 'var(--text-primary)',
                            border: '1px solid var(--border)',
                            borderRadius: '4px'
                        }}
                    >
                        ← Back to Image Grid
                    </button>
                    <h1>Feature Map Flow</h1>
                </div>

                <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem', flexWrap: 'wrap', width: '100%' }}>
                    <div style={{
                        background: 'var(--surface)',
                        padding: '0.5rem 1rem',
                        borderRadius: '6px',
                        fontSize: '0.85rem',
                        border: '1px solid var(--border)',
                        color: 'var(--text-secondary)',
                        flex: 1,
                        textAlign: 'center',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        whiteSpace: 'nowrap'
                    }}>
                        <strong>Epoch {epoch}</strong>&nbsp;• Image ID {imageId}
                    </div>

                    <div style={{
                        background: 'rgba(37, 99, 235, 0.1)',
                        padding: '0.5rem 1rem',
                        borderRadius: '6px',
                        fontSize: '0.85rem',
                        color: 'white',
                        border: '1px solid rgba(37, 99, 235, 0.2)',
                        flex: 1,
                        textAlign: 'center',
                        whiteSpace: 'nowrap'
                    }}>
                        Error Type: <strong>{errorType.charAt(0).toUpperCase() + errorType.slice(1)}</strong>
                    </div>

                    <div style={{
                        background: 'rgba(37, 99, 235, 0.1)',
                        padding: '0.5rem 1rem',
                        borderRadius: '6px',
                        fontSize: '0.85rem',
                        color: 'white',
                        border: '1px solid rgba(37, 99, 235, 0.2)',
                        flex: 1,
                        textAlign: 'center',
                        whiteSpace: 'nowrap'
                    }}>
                        Error Count: <strong>{passedErrorCount ?? imageErrorData.length}</strong>
                    </div>
                </div>
            </header>
            {/* <div className="image-detail-container">
                <div className="image-detail-grid">
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
            </div> */}

            <div className="image-detail-container">
                <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
                    {
                        imageErrorData.length > 0 ?
                            <><img
                                ref={imgRef}
                                src={resolveImagePath(imageId)}
                                alt={`Image ${imageId}`}
                                onLoad={drawBoundingBoxes}
                                style={{ display: 'block', maxWidth: '100%' }}
                                width={'100%'}
                            />
                                <canvas
                                    ref={canvasRef}
                                    style={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        maxWidth: '100%',
                                        height: 'auto',
                                        display: imageErrorData.length > 0 ? 'block' : 'none'
                                    }}
                                /></> :
                            <></>
                    }
                </div>
            </div>
        </div>
    );
}

export default ImageDetail;