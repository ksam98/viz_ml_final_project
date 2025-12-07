import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './ErrorBarChart.css';

const ErrorBarChart = ({ data, selectedEpoch, onEpochSelect, onErrorSelect, excludeErrors = [] }) => {
    const svgRef = useRef();

    useEffect(() => {
        if (!data || data.length === 0) return;

        // Clear previous chart
        d3.select(svgRef.current).selectAll('*').remove();

        // Prepare data for stacking
        // Structure: [{ epoch: 1, classification: 0.5, localization: 0.3, ... }, ...]
        const chartData = data.map(d => ({
            epoch: d.epoch,
            ...d.main_errors
        })).sort((a, b) => a.epoch - b.epoch);

        // Filter out excluded error types
        const allKeys = ['classification', 'localization', 'both', 'duplicate', 'background', 'miss'];
        const keys = allKeys.filter(key => !excludeErrors.includes(key));

        const colors = {
            classification: '#ff7f0e',
            localization: '#1f77b4',
            both: '#9467bd',
            duplicate: '#ffbb00',
            background: '#d62728',
            miss: '#7f7f7f'
        };

        // Dimensions
        const margin = { top: 20, right: 30, bottom: 40, left: 60 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;

        // Create SVG
        const svg = d3.select(svgRef.current)
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Scales
        const x = d3.scaleLinear()
            .domain(d3.extent(chartData, d => d.epoch))
            .range([0, width]);

        // Stack the data with filtered keys
        const stackedData = d3.stack()
            .keys(keys)
            (chartData);

        const y = d3.scaleLinear()
            .domain([0, d3.max(stackedData, layer => d3.max(layer, d => d[1])) * 1.1])
            .range([height, 0]);

        // Area generator
        const area = d3.area()
            .x(d => x(d.data.epoch))
            .y0(d => y(d[0]))
            .y1(d => y(d[1]))
            .curve(d3.curveMonotoneX); // Smooth curve

        // Axes
        // X Axis with integer ticks only
        const xAxis = d3.axisBottom(x)
            .ticks(chartData.length)
            .tickFormat(d3.format('d'));

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis)
            .style('font-size', '12px');

        // Y Axis
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5))
            .style('font-size', '12px');

        // Y-axis label
        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left + 15)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#666')
            .text('Stacked dAP');

        // X-axis label
        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('x', width / 2)
            .attr('y', height + margin.bottom - 5)
            .style('font-size', '12px')
            .style('fill', '#666')
            .text('Epoch');

        // Render Areas
        svg.selectAll('.layer')
            .data(stackedData)
            .enter()
            .append('path')
            .attr('class', 'layer')
            .attr('d', area)
            .style('fill', d => colors[d.key])
            .style('opacity', 0.8)
            .style('cursor', 'pointer')
            .on('mouseover', function (event, d) {
                d3.select(this).style('opacity', 1);
            })
            .on('mouseout', function (event, d) {
                d3.select(this).style('opacity', 0.8);
            })
            .on('click', function (event, d) {
                // d.key contains the error type (e.g., "classification")
                if (onErrorSelect) {
                    onErrorSelect(d.key);
                }

                // Prevent bubbling to the epoch selection click
                event.stopPropagation();
            });

        // Click handler for epoch selection (background/chart area)
        svg.on('click', function (event) {
            const [mouseX] = d3.pointer(event);
            const clickedEpoch = Math.round(x.invert(mouseX));

            // Clamp to valid range
            const validEpoch = Math.max(
                Math.min(clickedEpoch, d3.max(chartData, d => d.epoch)),
                d3.min(chartData, d => d.epoch)
            );

            if (onEpochSelect) {
                onEpochSelect(validEpoch);
            }
        });

        // Highlight Selected Epoch Line
        if (selectedEpoch) {
            svg.append('line')
                .attr('x1', x(selectedEpoch))
                .attr('x2', x(selectedEpoch))
                .attr('y1', 0)
                .attr('y2', height)
                .attr('stroke', '#333')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '5,5')
                .style('pointer-events', 'none'); // Let clicks pass through
        }

    }, [data, selectedEpoch, onEpochSelect, onErrorSelect, excludeErrors]);

    return <svg ref={svgRef} className="error-bar-chart"></svg>;
};

export default ErrorBarChart;