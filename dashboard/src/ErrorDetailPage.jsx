import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import * as d3 from 'd3';
import './App.css'; // Reuse main styles

import MetricsCard from './components/MetricsCard';
import './components/MetricsCard.css';

const ErrorDetailPage = () => {
    const { errorType } = useParams();
    const [searchParams] = useSearchParams();
    const epoch = Number(searchParams.get('epoch')) || 1;
    const navigate = useNavigate();

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [chartData, setChartData] = useState([]);

    useEffect(() => {
        setLoading(true);
        fetch(`/data/results_epoch_${epoch}.json`)
            .then(res => {
                if (!res.ok) throw new Error(`Failed to load data for Epoch ${epoch}`);
                return res.json();
            })
            .then(data => {
                processErrorData(data, errorType);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setError(err.message);
                setLoading(false);
            });
    }, [epoch, errorType]);

    const processErrorData = (data, type) => {
        // Map URL param type (lowercase) to JSON error code (e.g., "classification" -> "Cls")
        const typeMap = {
            'classification': 'Cls',
            'localization': 'Loc',
            'both': 'Both',
            'duplicate': 'Dupe',
            'background': 'Bkg',
            'miss': 'Miss'
        };

        const errorCode = typeMap[type.toLowerCase()];
        if (!errorCode) {
            setError(`Unknown error type: ${type}`);
            return;
        }

        // Aggregate errors per image
        // data.mispredictions is an array of objects: { image_id, errors: ["Cls", "Loc", ...] }
        const imageCounts = {};

        data.mispredictions.forEach(item => {
            const count = item.errors.filter(e => e === errorCode).length;
            if (count > 0) {
                imageCounts[item.image_id] = { count: count, boxes: item.boxes };
            }
        });

        // Convert to array for D3
        const processed = Object.entries(imageCounts).map(([id, value]) => ({
            id,
            value: value.count,
            boxes: value.boxes
        })).sort((a, b) => b.value - a.value); // Sort by count descending

        setChartData(processed);
    };

    if (loading) return <div className="loading-container"><div className="loading-spinner"></div></div>;
    if (error) return <div className="error-container"><h2>Error</h2><p>{error}</p><button onClick={() => navigate('/')}>Back to Dashboard</button></div>;

    const totalErrors = chartData.reduce((sum, item) => sum + item.value, 0);
    const affectedImages = chartData.length;

    return (
        <div className="app">
            <header className="app-header">
                <button
                    className="back-button"
                    onClick={() => navigate('/')}
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
                    ← Back to Dashboard
                </button>
                <h1>{errorType.charAt(0).toUpperCase() + errorType.slice(1)} Error Analysis</h1>

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
                        <strong>Epoch {epoch}</strong>&nbsp;• Per-Image Contribution
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
                        Total {errorType} errors in this epoch: <strong>{totalErrors}</strong>
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
                        Number of images containing this error: <strong>{affectedImages}</strong>
                    </div>
                </div>
            </header>

            <div className="viz-container">
                <div className="viz-card">
                    <h2>Image Contribution</h2>
                    <p className="viz-description">
                        Number of {errorType} errors contributed by each image.
                    </p>
                    <ImageContributionPieChart data={chartData} />
                </div>
                <div className="viz-card">
                    <h2>Image Grid</h2>
                    <ImageGrid data={chartData} epoch={epoch} errorType={errorType} />
                </div>
            </div>
        </div>
    );
};

const resolveImagePath = (imageId) => {
    const id_len = imageId.toString().length;
    const padding = '0'.repeat(12 - id_len);
    const imageFileName = padding + imageId.toString();
    return '/images/' + imageFileName + '.jpg';
}

const addBorderToSelectedImage = (imageId) => {
    const imgElement = document.getElementById(imageId);
    if (imgElement) {
        imgElement.style.border = '4px solid #2563eb'; // Tailwind blue-500
    }
}

const removeBorderFromSelectedImage = (imageId) => {
    const imgElement = document.getElementById(imageId);
    if (imgElement) {
        imgElement.style.border = 'none';
    }
}
const ImageGrid = ({ data, epoch, errorType }) => {
    console.log(data);
    const navigate = useNavigate();

    const handleImageClick = (imageId) => {
        // Navigate to ImageDetail page
        navigate(`/image/${imageId}/${errorType}?epoch=${epoch}`);
    };
    return (
        <div className="image-grid">
            {data.map(item => (
                <div
                    key={item.id}
                    className="image-card"
                    onClick={() => handleImageClick(item.id)}
                    style={{ cursor: 'pointer' }}
                >
                    <img
                        src={resolveImagePath(item.id)}
                        id={`${item.id}`}
                        alt={`Image ${item.id}`}
                    />
                </div>
            ))}
        </div>
    );
}

const ImageContributionPieChart = ({ data }) => {

    const ref = useRef();

    useEffect(() => {
        if (!data || data.length === 0) return;

        const width = 600;
        const height = 400;
        const radius = Math.min(width, height) / 2 - 40;

        d3.select(ref.current).selectAll('*').remove();

        const svg = d3.select(ref.current)
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width / 2},${height / 2})`);

        const color = d3.scaleOrdinal(d3.schemeCategory10);

        const pie = d3.pie()
            .value(d => d.value)
            .sort(null);

        const arc = d3.arc()
            .innerRadius(0)
            .outerRadius(radius);

        const labelArc = d3.arc()
            .innerRadius(radius + 10)
            .outerRadius(radius + 10);

        const arcs = svg.selectAll('arc')
            .data(pie(data))
            .enter()
            .append('g')
            .attr('class', 'arc');

        arcs.append('path')
            .attr('d', arc)
            .attr('fill', d => color(d.data.id))
            .attr('stroke', 'white')
            .style('stroke-width', '2px')
            .style('opacity', 0.8)
            .on('mouseover', function (event, d) {
                addBorderToSelectedImage(d.data.id)
                d3.select(this).style('opacity', 1);
            })
            .on('mouseout', function (event, d) {
                removeBorderFromSelectedImage(d.data.id)
                d3.select(this).style('opacity', 0.8);
            });

        // Add labels
        arcs.append('text')
            .attr('transform', d => `translate(${labelArc.centroid(d)})`)
            .attr('dy', '0.35em')
            .text(d => `Img ${d.data.id}`)
            .style('text-anchor', d => {
                const midAngle = d.startAngle + (d.endAngle - d.startAngle) / 2;
                return midAngle < Math.PI ? 'start' : 'end';
            })
            .style('font-size', '12px')
            .style('fill', '#cbd5e1');

        // Add values
        arcs.append('text')
            .attr('transform', d => `translate(${arc.centroid(d)})`)
            .attr('dy', '0.35em')
            .text(d => d.data.value)
            .style('text-anchor', 'middle')
            .style('fill', 'white')
            .style('font-weight', 'bold');

    }, [data]);

    return <svg ref={ref}></svg>;
};



export default ErrorDetailPage;
