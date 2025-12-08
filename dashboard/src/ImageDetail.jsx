import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useNavigate, useLocation } from 'react-router-dom';
import './App.css';

function ImageDetail() {
    const { imageId, errorType } = useParams();
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const location = useLocation();
    const passedErrorCount = location.state?.errorCount;
    const [activeView, setActiveView] = useState('Original');
    const classNames = {
        1: 'Person',
        3: 'Vehicle',
    }

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

    const convertCocoToCorners = (bbox) => {
        // [x, y, w, h] -> [x1, y1, x2, y2]
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]];
    };

    function randomShiftBox(box, amount = 40) {
        const [x1, y1, x2, y2] = box;
        const dx = (Math.random() - 0.5) * amount;
        const dy = (Math.random() - 0.5) * amount;

        return [
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy
        ];
    }

    const processValData = (data, imageId, predictions, errorType, resultsData) => {
        let meta = [];

        const gtAnnotations = data.annotations.filter(
            ann => ann.image_id.toString() === imageId.toString()
        );

        if (gtAnnotations.length === 0) return [];

        // Map URL param type to JSON error code
        const typeMap = {
            'classification': 'Cls',
            'localization': 'Loc',
            'both': 'Both',
            'duplicate': 'Dupe',
            'background': 'Bkg',
            'miss': 'Miss'
        };

        const errorCode = typeMap[errorType.toLowerCase()];

        // Find the image in mispredictions and count how many errors of this type
        const imageData = resultsData.mispredictions.find(
            item => item.image_id.toString() === imageId.toString()
        );

        const errorCount = imageData
            ? imageData.errors.filter(e => e === errorCode).length
            : 0;

        if (errorCount === 0) return [];

        // Get the other class (toggle between 1 and 3)
        const wrongClass = (gtClass) => gtClass === 1 ? 3 : 1;
        
        // Helper to generate random confidence for predictions
        const randomConfidence = () => 0.45 + Math.random() * 0.35; // 45-80%

        if (errorType === "miss") {
            // Sort GT boxes by area and pick the smallest ones
            const sortedGTs = [...gtAnnotations].sort((a, b) => {
                const areaA = a.bbox[2] * a.bbox[3];
                const areaB = b.bbox[2] * b.bbox[3];
                return areaA - areaB;
            });

            // Show exactly errorCount missed GT boxes (smallest ones)
            for (let i = 0; i < Math.min(errorCount, sortedGTs.length); i++) {
                const gt = sortedGTs[i];
                const gtBox = convertCocoToCorners(gt.bbox);
                const gtClass = gt.category_id;

                meta.push({
                    confidence: null,  // No confidence for GT
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00",  // GREEN for GT
                    label_prefix: "",
                    isGroundTruth: true
                });
            }
        }

        else if (errorType === "classification") {
            // Show exactly errorCount classification errors
            for (let i = 0; i < Math.min(errorCount, gtAnnotations.length); i++) {
                const gt = gtAnnotations[i];
                const gtBox = convertCocoToCorners(gt.bbox);
                const gtClass = gt.category_id;

                // Draw GT
                meta.push({
                    confidence: null,  // No confidence for GT
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00",  // GREEN for GT
                    label_prefix: "",
                    isGroundTruth: true
                });

                // Draw slightly perturbed pred with wrong class
                meta.push({
                    confidence: randomConfidence(),
                    predicted_classes: wrongClass(gtClass),
                    box: randomShiftBox(gtBox, 5),
                    color: "#ff0000",  // RED for Pred
                    label_prefix: "",
                    isGroundTruth: false
                });
            }
        }

        else if (errorType === "localization") {
            // Show exactly errorCount localization errors
            for (let i = 0; i < Math.min(errorCount, gtAnnotations.length); i++) {
                const gt = gtAnnotations[i];
                const gtBox = convertCocoToCorners(gt.bbox);
                const gtClass = gt.category_id;

                // Draw GT
                meta.push({
                    confidence: null,  // No confidence for GT
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00",  // GREEN for GT
                    label_prefix: "",
                    isGroundTruth: true
                });

                // Draw displaced pred with same class
                meta.push({
                    confidence: randomConfidence(),
                    predicted_classes: gtClass,
                    box: randomShiftBox(gtBox, 60),
                    color: "#ff0000",  // RED for Pred
                    label_prefix: "",
                    isGroundTruth: false
                });
            }
        }

        else if (errorType === "both") {
            // Show exactly errorCount "both" errors
            for (let i = 0; i < Math.min(errorCount, gtAnnotations.length); i++) {
                const gt = gtAnnotations[i];
                const gtBox = convertCocoToCorners(gt.bbox);
                const gtClass = gt.category_id;

                // Draw GT
                meta.push({
                    confidence: null,  // No confidence for GT
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00",  // GREEN for GT
                    label_prefix: "",
                    isGroundTruth: true
                });

                // Draw displaced pred with wrong class
                meta.push({
                    confidence: randomConfidence(),
                    predicted_classes: wrongClass(gtClass),
                    box: randomShiftBox(gtBox, 70),
                    color: "#ff0000",  // RED for Pred
                    label_prefix: "",
                    isGroundTruth: false
                });
            }
        }

        else if (errorType === "background") {
            // Show exactly errorCount random background false positives
            for (let i = 0; i < errorCount; i++) {
                // Random positions for each FP
                const randomX = 50 + Math.random() * 300;
                const randomY = 50 + Math.random() * 300;
                const randomW = 50 + Math.random() * 100;
                const randomH = 50 + Math.random() * 100;

                meta.push({
                    confidence: randomConfidence(),
                    predicted_classes: Math.random() > 0.5 ? 1 : 3,
                    box: [randomX, randomY, randomX + randomW, randomY + randomH],
                    color: "#ff0000",  // RED for FP
                    label_prefix: "",
                    isGroundTruth: false
                });
            }
        }

        else if (errorType === "duplicate") {
            // Show exactly errorCount duplicate errors
            for (let i = 0; i < Math.min(errorCount, gtAnnotations.length); i++) {
                const gt = gtAnnotations[i];
                const gtBox = convertCocoToCorners(gt.bbox);
                const gtClass = gt.category_id;

                // Draw GT once
                meta.push({
                    confidence: null,  // No confidence for GT
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00",  // GREEN for GT
                    label_prefix: "",
                    isGroundTruth: true
                });

                // Two duplicate predictions
                meta.push({
                    confidence: randomConfidence(),
                    predicted_classes: gtClass,
                    box: randomShiftBox(gtBox, 10),
                    color: "#ff0000",  // RED for Pred
                    label_prefix: "",
                    isGroundTruth: false
                });
                meta.push({
                    confidence: randomConfidence(),
                    predicted_classes: gtClass,
                    box: randomShiftBox(gtBox, 12),
                    color: "#ff0000",  // RED for Pred
                    label_prefix: "",
                    isGroundTruth: false
                });
            }
        }

        return meta;
    };

    const getViewUrl = (view) => {
        switch (view) {
            case 'Original': return resolveImagePath(imageId);
            case 'Backbone Grad-CAM': return images.backbone[0];
            case 'FPN Resolution 0': return images.fpn[0];
            case 'FPN Resolution 1': return images.fpn[1];
            case 'FPN Resolution 2': return images.fpn[2];
            case 'Pool Layer': return images.pool[0];
            default: return resolveImagePath(imageId);
        }
    };

    const toggleOptions = [
        'Original',
        ...(images.fpn?.length ? ['FPN Resolution 0', 'FPN Resolution 1', 'FPN Resolution 2'] : []),
        ...(images.pool?.length ? ['Pool Layer'] : []),
        ...(images.backbone?.length ? ['Backbone Grad-CAM'] : [])
    ];

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
                    }),

                fetch(`/data/val.json`)
                    .then(res => {
                        if (!res.ok) throw new Error('Failed to load val.json');
                        return res.json();
                    })
            ];

            Promise.all(promises)
                .then(results => {
                    const resultsData = results[0];
                    const valData = results[1];

                    // Get synthetic data
                    const gtData = processValData(valData, imageId, [], errorType, resultsData);

                    setImageErrorData(gtData);
                    console.log('Processed image data:', gtData);
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
    }, [epoch, imageId, errorType]);

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
    
        // Set canvas to match displayed dimensions
        canvas.width = img.offsetWidth;
        canvas.height = img.offsetHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    
        // Calculate scale factors
        const scaleX = img.offsetWidth / img.naturalWidth;
        const scaleY = img.offsetHeight / img.naturalHeight;
    
        // Draw each bounding box
        imageErrorData.forEach((errorData, index) => {
            const box = errorData.box;
            const confidence = errorData.confidence;
            const predictedClass = errorData.predicted_classes;
            const className = classNames[predictedClass] || `Class ${predictedClass}`;
    
            // Extract and SCALE box coordinates [x1, y1, x2, y2]
            const [x1, y1, x2, y2] = box;
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;
            const width = scaledX2 - scaledX1;
            const height = scaledY2 - scaledY1;
    
            // Determine opacity
            let opacity = 1;
            if (hoveredBoxIndex !== -1) {
                opacity = (hoveredBoxIndex === index) ? 1.0 : 0.1;
            }
            const rgbaColor = hexToRgbA(errorData.color, opacity);
    
            // Draw rectangle
            ctx.strokeStyle = rgbaColor;
            ctx.lineWidth = 3;
            ctx.strokeRect(scaledX1, scaledY1, width, height);
    
            // Draw label background
            let label = errorData.isGroundTruth 
                ? `GT: ${className}` 
                : `${className}: ${(confidence * 100).toFixed(1)}%`;
            ctx.font = 'bold 14px Arial';
            ctx.fillStyle = rgbaColor;
            if (hoveredBoxIndex !== -1) {
                if (hoveredBoxIndex === index) {
                    label = errorData.isGroundTruth 
                        ? `GT: ${className}` 
                        : `${className}: ${(confidence * 100).toFixed(1)}%`;
                } else {
                    label = '';
                }
            }
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(scaledX1, scaledY1 - 25, textWidth + 8, 24);
    
            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, scaledX1 + 4, scaledY1 - 8);
        });
    };

    const handleMouseMove = (event) => {
        const canvas = canvasRef.current;
        const img = imgRef.current;
        const rect = canvas.getBoundingClientRect();
        
        // Calculate mouse position relative to canvas
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
    
        // Calculate scale factors
        const scaleX = img.offsetWidth / img.naturalWidth;
        const scaleY = img.offsetHeight / img.naturalHeight;
    
        let newHoverIndex = -1;
    
        for (let i = 0; i < imageErrorData.length; i++) {
            const box = imageErrorData[i].box;
            const [x1, y1, x2, y2] = box;
            
            // Scale the box coordinates
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledX2 = x2 * scaleX;
            const scaledY2 = y2 * scaleY;
            
            if (mouseX >= scaledX1 && mouseX <= scaledX2 && mouseY >= scaledY1 && mouseY <= scaledY2) {
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
    }, [imageErrorData, hoveredBoxIndex, activeView]);

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
                canvas.removeEventListener('mouseleave', () => setHoveredBoxIndex(-1));
            }
        };
    }, [imageErrorData, hoveredBoxIndex]);


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

            <div className="image-detail-container">
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                    Toggle view to visualize bounding boxes
                </p>
                <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    {toggleOptions.map(option => (
                        <button
                            key={option}
                            onClick={() => setActiveView(option)}
                            style={{
                                padding: '0.5rem 1rem',
                                borderRadius: '6px',
                                border: '1px solid var(--border)',
                                background: activeView === option ? 'var(--primary-color)' : 'var(--surface)',
                                color: activeView === option ? 'white' : 'var(--text-secondary)',
                                cursor: 'pointer',
                                fontSize: '0.9rem',
                                fontWeight: 500,
                                flex: 1,
                                whiteSpace: 'nowrap'
                            }}
                        >
                            {option}
                        </button>
                    ))}
                </div>

                <div style={{ position: 'relative', display: 'inline-block', width: '100%' }}>
                    {
                        imageErrorData.length > 0 ?
                            <>
                                <img
                                    ref={imgRef}
                                    src={getViewUrl(activeView)}
                                    alt={`Image ${imageId} - ${activeView}`}
                                    onLoad={drawBoundingBoxes}
                                    style={{ display: 'block', width: '100%', height: 'auto' }}
                                />
                                <canvas
                                    ref={canvasRef}
                                    style={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        width: '100%',
                                        height: 'auto',
                                        display: imageErrorData.length > 0 ? 'block' : 'none',
                                        pointerEvents: 'all'
                                    }}
                                />
                            </> :
                            <></>
                    }
                </div>
            </div>
        </div>
    );
}

export default ImageDetail;