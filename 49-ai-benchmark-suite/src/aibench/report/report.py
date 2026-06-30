"""Benchmark reporting and visualization."""

from dataclasses import dataclass, field
from typing import Any
import json
import os
from pathlib import Path
import time

from ..core.benchmark import BenchmarkResult, Metric, MetricType


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: str = "./reports"
    format: str = "html"  # html, markdown, json
    include_charts: bool = True
    include_raw_data: bool = False
    title: str = "AI Benchmark Report"


class ReportGenerator:
    """Generate benchmark reports."""

    def __init__(self, config: ReportConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: list[BenchmarkResult],
        filename: str | None = None
    ) -> str:
        """Generate report from results."""
        if self.config.format == "html":
            return self._generate_html(results, filename)
        elif self.config.format == "markdown":
            return self._generate_markdown(results, filename)
        elif self.config.format == "json":
            return self._generate_json(results, filename)
        else:
            return self._generate_markdown(results, filename)

    def _generate_html(
        self,
        results: list[BenchmarkResult],
        filename: str | None
    ) -> str:
        """Generate HTML report."""
        if filename is None:
            filename = f"report_{int(time.time())}.html"

        filepath = os.path.join(self.config.output_dir, filename)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.config.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .metric {{ font-weight: bold; }}
        .chart {{ margin: 20px 0; padding: 20px; background: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>{self.config.title}</h1>
    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Summary</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Status</th>
            <th>Mean (ms)</th>
            <th>Std (ms)</th>
            <th>P99 (ms)</th>
        </tr>
"""

        for result in results:
            status_class = "pass" if result.success else "fail"
            status_text = "PASS" if result.success else "FAIL"

            html += f"""        <tr>
            <td>{result.benchmark_name}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{result.mean_time_ms:.2f}</td>
            <td>{result.std_time_ms:.2f}</td>
            <td>{result.p99_time_ms:.2f}</td>
        </tr>
"""

        html += """    </table>

    <h2>Detailed Results</h2>
"""

        for result in results:
            html += f"""    <h3>{result.benchmark_name}</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Unit</th></tr>
"""
            for metric in result.metrics:
                html += f"""        <tr>
            <td>{metric.name}</td>
            <td class="metric">{metric.value:.4f}</td>
            <td>{metric.unit.value}</td>
        </tr>
"""
            html += """    </table>
"""

        if self.config.include_charts:
            html += self._generate_ascii_chart(results)

        html += """</body>
</html>"""

        with open(filepath, 'w') as f:
            f.write(html)

        return filepath

    def _generate_markdown(
        self,
        results: list[BenchmarkResult],
        filename: str | None
    ) -> str:
        """Generate Markdown report."""
        if filename is None:
            filename = f"report_{int(time.time())}.md"

        filepath = os.path.join(self.config.output_dir, filename)

        md = f"""# {self.config.title}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Benchmark | Status | Mean (ms) | Std (ms) | P99 (ms) |
|-----------|--------|-----------|----------|----------|
"""

        for result in results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            md += f"| {result.benchmark_name} | {status} | {result.mean_time_ms:.2f} | {result.std_time_ms:.2f} | {result.p99_time_ms:.2f} |\n"

        md += "\n## Detailed Results\n\n"

        for result in results:
            md += f"### {result.benchmark_name}\n\n"

            if not result.success:
                md += f"**Error:** {result.error_message}\n\n"
                continue

            md += "| Metric | Value | Unit |\n"
            md += "|--------|-------|------|\n"

            for metric in result.metrics:
                md += f"| {metric.name} | {metric.value:.4f} | {metric.unit.value} |\n"

            md += "\n"

        with open(filepath, 'w') as f:
            f.write(md)

        return filepath

    def _generate_json(
        self,
        results: list[BenchmarkResult],
        filename: str | None
    ) -> str:
        """Generate JSON report."""
        if filename is None:
            filename = f"report_{int(time.time())}.json"

        filepath = os.path.join(self.config.output_dir, filename)

        data = {
            "title": self.config.title,
            "generated": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": [r.to_dict() for r in results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def _generate_ascii_chart(self, results: list[BenchmarkResult]) -> str:
        """Generate ASCII bar chart for HTML."""
        if not results:
            return ""

        max_time = max(r.mean_time_ms for r in results if r.success)
        if max_time == 0:
            return ""

        chart = """    <h2>Performance Chart</h2>
    <div class="chart">
        <pre>
"""

        for result in results:
            if not result.success:
                continue

            bar_length = int((result.mean_time_ms / max_time) * 40)
            bar = "█" * bar_length
            name = result.benchmark_name[:20].ljust(20)
            chart += f"{name} |{bar} {result.mean_time_ms:.1f}ms\n"

        chart += """        </pre>
    </div>
"""

        return chart


class ComparisonReport:
    """Generate comparison reports between runs."""

    def __init__(self, config: ReportConfig):
        self.config = config

    def compare(
        self,
        baseline: list[BenchmarkResult],
        current: list[BenchmarkResult],
        filename: str | None = None
    ) -> str:
        """Generate comparison report."""
        if filename is None:
            filename = f"comparison_{int(time.time())}.md"

        filepath = os.path.join(self.config.output_dir, filename)

        baseline_by_name = {r.benchmark_name: r for r in baseline}
        current_by_name = {r.benchmark_name: r for r in current}

        md = f"""# Benchmark Comparison Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Benchmark | Baseline (ms) | Current (ms) | Speedup | Status |
|-----------|---------------|--------------|---------|--------|
"""

        for name in baseline_by_name:
            if name in current_by_name:
                base = baseline_by_name[name]
                curr = current_by_name[name]

                if base.mean_time_ms > 0:
                    speedup = base.mean_time_ms / curr.mean_time_ms
                else:
                    speedup = 1.0

                if speedup > 1.05:
                    status = "🟢 Improved"
                elif speedup < 0.95:
                    status = "🔴 Regressed"
                else:
                    status = "🟡 Same"

                md += f"| {name} | {base.mean_time_ms:.2f} | {curr.mean_time_ms:.2f} | {speedup:.2f}x | {status} |\n"

        md += "\n## Analysis\n\n"

        # Find improvements and regressions
        improvements = []
        regressions = []

        for name in baseline_by_name:
            if name in current_by_name:
                base = baseline_by_name[name]
                curr = current_by_name[name]
                if base.mean_time_ms > 0:
                    speedup = base.mean_time_ms / curr.mean_time_ms
                    if speedup > 1.05:
                        improvements.append((name, speedup))
                    elif speedup < 0.95:
                        regressions.append((name, speedup))

        if improvements:
            md += "### Improvements\n\n"
            for name, speedup in sorted(improvements, key=lambda x: -x[1]):
                md += f"- **{name}**: {speedup:.2f}x faster\n"
            md += "\n"

        if regressions:
            md += "### Regressions\n\n"
            for name, speedup in sorted(regressions, key=lambda x: x[1]):
                md += f"- **{name}**: {speedup:.2f}x slower\n"
            md += "\n"

        with open(filepath, 'w') as f:
            f.write(md)

        return filepath


class MetricsAnalyzer:
    """Analyze benchmark metrics."""

    def analyze(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Analyze results and provide insights."""
        analysis = {
            "total_benchmarks": len(results),
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "metrics_summary": {}
        }

        # Aggregate metrics by type
        metric_values: dict[str, list[float]] = {}

        for result in results:
            for metric in result.metrics:
                key = metric.name
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(metric.value)

        # Calculate statistics
        for name, values in metric_values.items():
            if values:
                analysis["metrics_summary"][name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        # Identify bottlenecks
        analysis["bottlenecks"] = self._identify_bottlenecks(results)

        # Recommendations
        analysis["recommendations"] = self._generate_recommendations(results)

        return analysis

    def _identify_bottlenecks(
        self,
        results: list[BenchmarkResult]
    ) -> list[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        # Find slowest benchmarks
        sorted_results = sorted(
            [r for r in results if r.success],
            key=lambda r: r.mean_time_ms,
            reverse=True
        )

        if sorted_results:
            slowest = sorted_results[0]
            if slowest.mean_time_ms > 100:  # > 100ms
                bottlenecks.append(f"Slow benchmark: {slowest.benchmark_name} ({slowest.mean_time_ms:.2f}ms)")

        # Check for high variance
        for result in results:
            if result.success and result.std_time_ms > result.mean_time_ms * 0.2:
                bottlenecks.append(f"High variance: {result.benchmark_name}")

        return bottlenecks

    def _generate_recommendations(
        self,
        results: list[BenchmarkResult]
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        for result in results:
            if not result.success:
                recommendations.append(f"Fix failing benchmark: {result.benchmark_name}")
                continue

            # Check memory usage
            for metric in result.metrics:
                if metric.name == "peak_memory" and metric.value > 8000:  # > 8GB
                    recommendations.append(
                        f"Optimize memory for {result.benchmark_name} (using {metric.value:.0f}MB)"
                    )

        return recommendations


def generate_report(
    results: list[BenchmarkResult],
    output_dir: str = "./reports",
    format: str = "markdown"
) -> str:
    """Convenience function to generate report."""
    config = ReportConfig(output_dir=output_dir, format=format)
    generator = ReportGenerator(config)
    return generator.generate(results)
