import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import './App.css';

function ImageDetail() {
    const { imageId, errorType } = useParams();
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const epoch = searchParams.get('epoch') || '1';

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
    };

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
                    confidence: 1,
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00ff",  // GREEN for GT
                    label_prefix: ""
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
                    confidence: 1,
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00ff",  // GREEN for GT
                    label_prefix: ""
                });
    
                // Draw slightly perturbed pred with wrong class
                meta.push({
                    confidence: 0.5,
                    predicted_classes: wrongClass(gtClass),
                    box: randomShiftBox(gtBox, 5),
                    color: "#ff0000ff",  // RED for Pred
                    label_prefix: ""
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
                    confidence: 1,
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00ff",  // GREEN for GT
                    label_prefix: ""
                });
    
                // Draw displaced pred with same class
                meta.push({
                    confidence: 0.7,
                    predicted_classes: gtClass,
                    box: randomShiftBox(gtBox, 60),
                    color: "#ff0000ff",  // RED for Pred
                    label_prefix: ""
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
                    confidence: 1,
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00ff",  // GREEN for GT
                    label_prefix: ""
                });
    
                // Draw displaced pred with wrong class
                meta.push({
                    confidence: 0.4,
                    predicted_classes: wrongClass(gtClass),
                    box: randomShiftBox(gtBox, 70),
                    color: "#ff0000ff",  // RED for Pred
                    label_prefix: ""
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
                    confidence: 0.3 + Math.random() * 0.3,
                    predicted_classes: Math.random() > 0.5 ? 1 : 3,
                    box: [randomX, randomY, randomX + randomW, randomY + randomH],
                    color: "#ff0000ff",  // RED for FP
                    label_prefix: ""
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
                    confidence: 1,
                    predicted_classes: gtClass,
                    box: gtBox,
                    color: "#00ff00ff",  // GREEN for GT
                    label_prefix: ""
                });
    
                // Two duplicate predictions
                meta.push({
                    confidence: 0.6,
                    predicted_classes: gtClass,
                    box: randomShiftBox(gtBox, 10),
                    color: "#ff0000ff",  // RED for Pred
                    label_prefix: ""
                });
                meta.push({
                    confidence: 0.55,
                    predicted_classes: gtClass,
                    box: randomShiftBox(gtBox, 12),
                    color: "#ff0000ff",  // RED for Pred
                    label_prefix: ""
                });
            }
        }
    
        return meta;
    };
    
    const processData = (data, imageId, errorType) => {
        // Not needed anymore since we're using processValData for everything
        return [];
    };

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
                    
                    // First get predictions
                    const predData = processData(resultsData, imageId, errorType);
                    
                    // Then get matching GT boxes
                    const gtData = processValData(valData, imageId, [], errorType, resultsData);
                    
                    setImageErrorData([...predData, ...gtData]);
                    console.log('Processed image data:', [...predData, ...gtData]);
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

    const drawBoundingBoxes = () => {
        if (!imgRef.current || !canvasRef.current || imageErrorData.length === 0) return;

        const canvas = canvasRef.current;
        const img = imgRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas dimensions to match image
        canvas.width = img.width;
        canvas.height = img.height;

        // Draw each bounding box
        imageErrorData.forEach(errorData => {
            const box = errorData.box;
            const confidence = errorData.confidence;
            const predictedClass = errorData.predicted_classes;
            const className = `Class ${predictedClass}`;
            const prefix = errorData.label_prefix || '';

            // Extract box coordinates [x1, y1, x2, y2]
            const [x1, y1, x2, y2] = box;
            const width = x2 - x1;
            const height = y2 - y1;

            // Draw rectangle
            ctx.strokeStyle = errorData.color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, width, height);

            // Draw label background
            const label = prefix === 'GT' || prefix.includes('GT')
                ? `${prefix}: ${className}` 
                : `${prefix}: ${className} ${(confidence * 100).toFixed(1)}%`;
            
            ctx.font = 'bold 14px Arial';
            ctx.fillStyle = errorData.color;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x1, y1 - 25, textWidth + 8, 24);

            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x1 + 4, y1 - 8);
        });
    };

    useEffect(() => {
        if (imgRef.current && imgRef.current.complete) {
            drawBoundingBoxes();
        }
    }, [imageErrorData]);
    
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
            <div className="image-detail-container">
                <h2>Error Type: {errorType.charAt(0).toUpperCase() + errorType.slice(1)}</h2>
                <div style={{ position: 'relative', display: 'inline-block' }}>
                    {
                        imageErrorData.length > 0 ?
                            <><img 
                                ref={imgRef}
                                src={resolveImagePath(imageId)} 
                                alt={`Image ${imageId}`}
                                onLoad={drawBoundingBoxes}
                                style={{ display: 'block', maxWidth: '100%' }}
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
                            /></>:
                            <></>
                    }
                </div>
            </div>
        </div>
    );
}

export default ImageDetail;