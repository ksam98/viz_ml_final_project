import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import './ErrorBarChart.css';

const ErrorBarChart = ({ data }) => {
    const svgRef = useRef();

    useEffect(() => {
        if (!data) return;

        // Clear previous chart
        d3.select(svgRef.current).selectAll('*').remove();

        // Prepare data
        const errorData = [
            { type: 'Classification', value: data.classification, color: '#ff7f0e' },
            { type: 'Localization', value: data.localization, color: '#1f77b4' },
            { type: 'Both', value: data.both, color: '#9467bd' },
            { type: 'Duplicate', value: data.duplicate, color: '#ffbb00' },
            { type: 'Background', value: data.background, color: '#d62728' },
            { type: 'Miss', value: data.miss, color: '#7f7f7f' }
        ];

        // Dimensions
        const margin = { top: 20, right: 30, bottom: 60, left: 80 };
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
            .domain(errorData.map(d => d.type))
            .range([0, width])
            .padding(0.2);

        const y = d3.scaleLinear()
            .domain([0, d3.max(errorData, d => d.value) * 1.1])
            .nice()
            .range([height, 0]);

        // Axes
        svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x))
            .selectAll('text')
            .attr('transform', 'rotate(-45)')
            .style('text-anchor', 'end');

        svg.append('g')
            .attr('class', 'y-axis')
            .call(d3.axisLeft(y).ticks(5).tickFormat(d => (d * 100).toFixed(1) + '%'));

        // Y-axis label
        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left + 20)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#666')
            .text('Impact on AP (dAP)');

        // Bars
        const bars = svg.selectAll('.bar')
            .data(errorData)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', d => x(d.type))
            .attr('width', x.bandwidth())
            .attr('y', height)
            .attr('height', 0)
            .attr('fill', d => d.color)
            .attr('rx', 4);

        // Animate bars
        bars.transition()
            .duration(800)
            .attr('y', d => y(d.value))
            .attr('height', d => height - y(d.value));

        // Tooltips
        const tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        bars.on('mouseover', function (event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('opacity', 0.7);

            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
            tooltip.html(`
        <strong>${d.type}</strong><br/>
        dAP: ${(d.value * 100).toFixed(2)}%
      `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
            .on('mouseout', function () {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('opacity', 1);

                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
            });

        return () => {
            tooltip.remove();
        };
    }, [data]);

    return <svg ref={svgRef} className="error-bar-chart"></svg>;
};

export default ErrorBarChart;
