import './MetricsCard.css';

const MetricsCard = ({ title, value, unit = '', description }) => {
    return (
        <div className="metrics-card">
            <h3 className="metrics-title">{title}</h3>
            <div className="metrics-value">
                {value}{unit}
            </div>
            <p className="metrics-description">{description}</p>
        </div>
    );
};

export default MetricsCard;
