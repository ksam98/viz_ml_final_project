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
        const x = d3.scaleBand()
            .domain(chartData.map(d => d.epoch))
            .range([0, width])
            .padding(0.3);

        // Stack the data with filtered keys
        const stackedData = d3.stack()
            .keys(keys)
            (chartData);

        const y = d3.scaleLinear()
            .domain([0, d3.max(stackedData, layer => d3.max(layer, d => d[1])) * 1.1])
            .range([height, 0]);

        // Axes
        const xAxis = d3.axisBottom(x)
            .tickFormat(d => `Epoch ${d}`);

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

        // Create tooltip
        const tooltip = d3.select('body')
            .selectAll('.chart-tooltip')
            .data([null])
            .join('div')
            .attr('class', 'chart-tooltip')
            .style('opacity', 0);

        // Render Bars
        svg.selectAll('.layer')
            .data(stackedData)
            .enter()
            .append('g')
            .attr('class', 'layer')
            .attr('fill', d => colors[d.key])
            .selectAll('rect')
            .data(d => d.map(item => ({ ...item, key: d.key }))) // Pass key down to rects
            .enter()
            .append('rect')
            .attr('x', d => x(d.data.epoch))
            .attr('y', d => y(d[1]))
            .attr('height', d => y(d[0]) - y(d[1]))
            .attr('width', x.bandwidth())
            .style('opacity', 0.9)
            .on('mouseover', function (event, d) {
                d3.select(this).style('opacity', 1).attr('stroke', '#fff').attr('stroke-width', 1);

                const dAP = (d[1] - d[0]).toFixed(4);
                tooltip.transition().duration(200).style('opacity', 0.9);
                tooltip.html(`
                    <h4>Epoch ${d.data.epoch}</h4>
                    <p><strong>Error:</strong> <span style="color: ${colors[d.key]}">${d.key.charAt(0).toUpperCase() + d.key.slice(1)}</span></p>
                    <p><strong>dAP:</strong> ${dAP}</p>
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mousemove', function (event) {
                tooltip
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', function (event, d) {
                d3.select(this).style('opacity', 0.9).attr('stroke', 'none');
                tooltip.transition().duration(500).style('opacity', 0);
            })
            .on('click', function (event, d) {
                // d.key contains the error type (e.g., "classification")
                if (onErrorSelect) {
                    onErrorSelect(d.key);
                }

                // Also select the epoch
                if (onEpochSelect) {
                    onEpochSelect(d.data.epoch);
                }

                event.stopPropagation();
            });

        return () => {
            tooltip.remove();
        };
    }, [data, selectedEpoch, onEpochSelect, onErrorSelect, excludeErrors]);

    return <svg ref={svgRef} className="error-bar-chart"></svg>;
};

export default ErrorBarChart;