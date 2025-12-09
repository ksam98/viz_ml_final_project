import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const ErrorDetailBarChart = ({ data, onErrorClick }) => {
    const svgRef = useRef();

    useEffect(() => {
        if (!data) return;

        // Clear previous chart
        d3.select(svgRef.current).selectAll('*').remove();

        // Prepare data - exclude background
        const barData = [
            { type: 'classification', label: 'Classification', value: data.classification, color: '#ff7f0e' },
            { type: 'localization', label: 'Localization', value: data.localization, color: '#1f77b4' },
            { type: 'both', label: 'Both', value: data.both, color: '#9467bd' },
            { type: 'duplicate', label: 'Duplicate', value: data.duplicate, color: '#ffbb00' },
            { type: 'miss', label: 'Miss', value: data.miss, color: '#7f7f7f' }
        ].filter(d => d.value > 0)
        .sort((a, b) => b.value - a.value);

        if (barData.length === 0) return;

        // Dimensions
        const margin = { top: 20, right: 20, bottom: 70, left: 70 };
        const width = 550 - margin.left - margin.right;
        const height = 350 - margin.top - margin.bottom;

        // Create SVG
        const svg = d3.select(svgRef.current)
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Scales
        const x = d3.scaleBand()
            .domain(barData.map(d => d.label))
            .range([0, width])
            .padding(0.3);

        const y = d3.scaleLinear()
            .domain([0, d3.max(barData, d => d.value) * 1.15])
            .range([height, 0]);

        // Axes
        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x))
            .selectAll('text')
            .style('text-anchor', 'end')
            .attr('dx', '-.8em')
            .attr('dy', '.15em')
            .attr('transform', 'rotate(-45)')
            .style('font-size', '12px')
            .style('fill', '#cbd5e1');

        svg.append('g')
            .call(d3.axisLeft(y).ticks(5))
            .style('font-size', '12px')
            .selectAll('text')
            .style('fill', '#cbd5e1');

        // Axis lines
        svg.selectAll('.domain, .tick line')
            .style('stroke', '#334155');

        // Y-axis label
        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left + 20)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#94a3b8')
            .text('dAP Impact');

        // Tooltip
        const tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        // Bars
        const bars = svg.selectAll('.bar')
            .data(barData)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', d => x(d.label))
            .attr('width', x.bandwidth())
            .attr('y', height)
            .attr('height', 0)
            .attr('fill', d => d.color)
            .attr('rx', 4)
            .style('cursor', 'pointer')
            .style('opacity', 0.85);

        // Animate bars
        bars.transition()
            .duration(800)
            .delay((d, i) => i * 100)
            .attr('y', d => y(d.value))
            .attr('height', d => height - y(d.value));

        // Add value labels on top of bars
        svg.selectAll('.bar-label')
            .data(barData)
            .enter()
            .append('text')
            .attr('class', 'bar-label')
            .attr('x', d => x(d.label) + x.bandwidth() / 2)
            .attr('y', height)
            .attr('text-anchor', 'middle')
            .style('font-size', '11px')
            .style('font-weight', 'bold')
            .style('fill', '#f1f5f9')
            .style('opacity', 0)
            .text(d => (d.value * 100).toFixed(1))
            .transition()
            .duration(800)
            .delay((d, i) => i * 100)
            .attr('y', d => y(d.value) - 5)
            .style('opacity', 1);

        // Interactions
        bars.on('mouseover', function (event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .style('opacity', 1)
                .attr('y', y(d.value) - 5)
                .attr('height', height - y(d.value) + 5);

            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
            tooltip.html(`
                <strong>${d.label}</strong><br/>
                dAP: ${(d.value * 100).toFixed(2)}%<br/>
                <em style="font-size: 0.85em; color: #94a3b8;">Click to view details</em>
            `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function (event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .style('opacity', 0.85)
                .attr('y', y(d.value))
                .attr('height', height - y(d.value));

            tooltip.transition()
                .duration(500)
                .style('opacity', 0);
        })
        .on('click', function (event, d) {
            if (onErrorClick) {
                onErrorClick(d.type);
            }
        });

        return () => {
            tooltip.remove();
        };
    }, [data, onErrorClick]);

    return <svg ref={svgRef} style={{ display: 'block', margin: '0 auto' }}></svg>;
};

export default ErrorDetailBarChart;