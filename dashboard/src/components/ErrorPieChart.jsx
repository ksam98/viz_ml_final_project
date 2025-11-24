import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './ErrorPieChart.css';

const ErrorPieChart = ({ data }) => {
    const svgRef = useRef();

    useEffect(() => {
        if (!data) return;

        // Clear previous chart
        d3.select(svgRef.current).selectAll('*').remove();

        // Prepare data - filter out zero values
        const errorData = [
            { type: 'Classification', value: data.classification, color: '#ff7f0e' },
            { type: 'Localization', value: data.localization, color: '#1f77b4' },
            { type: 'Both', value: data.both, color: '#9467bd' },
            { type: 'Duplicate', value: data.duplicate, color: '#ffbb00' },
            { type: 'Background', value: data.background, color: '#d62728' },
            { type: 'Miss', value: data.miss, color: '#7f7f7f' }
        ].filter(d => d.value > 0);

        if (errorData.length === 0) {
            return;
        }

        // Dimensions
        const width = 400;
        const height = 400;
        const radius = Math.min(width, height) / 2 - 40;

        // Create SVG
        const svg = d3.select(svgRef.current)
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width / 2},${height / 2})`);

        // Create pie layout
        const pie = d3.pie()
            .value(d => d.value)
            .sort(null);

        // Create arc
        const arc = d3.arc()
            .innerRadius(radius * 0.5)
            .outerRadius(radius);

        const arcHover = d3.arc()
            .innerRadius(radius * 0.5)
            .outerRadius(radius * 1.1);

        // Create slices
        const slices = svg.selectAll('path')
            .data(pie(errorData))
            .enter()
            .append('path')
            .attr('d', arc)
            .attr('fill', d => d.data.color)
            .attr('stroke', 'white')
            .attr('stroke-width', 2)
            .style('opacity', 0);

        // Animate slices
        slices.transition()
            .duration(800)
            .style('opacity', 1)
            .attrTween('d', function (d) {
                const interpolate = d3.interpolate({ startAngle: 0, endAngle: 0 }, d);
                return function (t) {
                    return arc(interpolate(t));
                };
            });

        // Add labels
        const labels = svg.selectAll('text')
            .data(pie(errorData))
            .enter()
            .append('text')
            .attr('transform', d => `translate(${arc.centroid(d)})`)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .style('fill', 'white')
            .style('opacity', 0)
            .text(d => {
                const percentage = (d.data.value / d3.sum(errorData, d => d.value) * 100);
                return percentage > 5 ? `${percentage.toFixed(0)}%` : '';
            });

        labels.transition()
            .delay(800)
            .duration(400)
            .style('opacity', 1);

        // Tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        slices.on('mouseover', function (event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('d', arcHover);

            const percentage = (d.data.value / d3.sum(errorData, d => d.value) * 100);
            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
            tooltip.html(`
        <strong>${d.data.type}</strong><br/>
        dAP: ${(d.data.value * 100).toFixed(2)}%<br/>
        ${percentage.toFixed(1)}% of total errors
      `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
            .on('mouseout', function () {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('d', arc);

                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
            });

        // Legend
        const legend = svg.selectAll('.legend')
            .data(errorData)
            .enter()
            .append('g')
            .attr('class', 'legend')
            .attr('transform', (d, i) => `translate(${radius + 20}, ${-radius + i * 25})`);

        legend.append('rect')
            .attr('width', 18)
            .attr('height', 18)
            .attr('fill', d => d.color);

        legend.append('text')
            .attr('x', 24)
            .attr('y', 9)
            .attr('dy', '.35em')
            .style('font-size', '12px')
            .text(d => d.type);

        return () => {
            tooltip.remove();
        };
    }, [data]);

    return <svg ref={svgRef} className="error-pie-chart"></svg>;
};

export default ErrorPieChart;
