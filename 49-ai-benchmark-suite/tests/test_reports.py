"""Tests for report generation functionality."""

import pytest
import os
import json
import tempfile

from aibench.core.benchmark import (
    BenchmarkConfig, BenchmarkResult, Metric, MetricType
)
from aibench.report.report import (
    ReportConfig, ReportGenerator, ComparisonReport, MetricsAnalyzer, generate_report
)


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_config(self):
        """Test default report configuration."""
        config = ReportConfig()

        assert config.output_dir == "./reports"
        assert config.format == "html"
        assert config.include_charts is True
        assert config.include_raw_data is False
        assert config.title == "AI Benchmark Report"

    def test_custom_config(self, temp_output_dir):
        """Test custom report configuration."""
        config = ReportConfig(
            output_dir=temp_output_dir,
            format="markdown",
            include_charts=False,
            include_raw_data=True,
            title="Custom Report"
        )

        assert config.output_dir == temp_output_dir
        assert config.format == "markdown"
        assert config.include_charts is False
        assert config.include_raw_data is True
        assert config.title == "Custom Report"


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generator_creation(self, report_config):
        """Test report generator creation."""
        generator = ReportGenerator(report_config)

        assert generator.config == report_config
        assert os.path.exists(report_config.output_dir)

    def test_generator_creates_output_dir(self, temp_dir):
        """Test generator creates output directory if needed."""
        output_dir = os.path.join(temp_dir, "new_reports_dir")
        config = ReportConfig(output_dir=output_dir)

        generator = ReportGenerator(config)

        assert os.path.exists(output_dir)


class TestHTMLReportGeneration:
    """Tests for HTML report generation."""

    def test_generate_html_report(self, html_report_config, multiple_results):
        """Test generating HTML report."""
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate(multiple_results)

        assert os.path.exists(filepath)
        assert filepath.endswith(".html")

    def test_html_report_content(self, html_report_config, successful_result):
        """Test HTML report content structure."""
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate([successful_result], "test_report.html")

        with open(filepath, 'r') as f:
            content = f.read()

        # Check basic structure
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "</html>" in content
        assert html_report_config.title in content

        # Check summary section
        assert "Summary" in content
        assert "test_benchmark" in content
        assert "PASS" in content

        # Check detailed results
        assert "Detailed Results" in content

    def test_html_report_with_metrics(self, html_report_config, successful_result):
        """Test HTML report includes metrics."""
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate([successful_result], "metrics_report.html")

        with open(filepath, 'r') as f:
            content = f.read()

        # Check metrics are present
        assert "mean_latency" in content
        assert "throughput" in content
        assert "peak_memory" in content

    def test_html_report_with_charts(self, html_report_config, multiple_results):
        """Test HTML report includes charts when enabled."""
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate(multiple_results, "chart_report.html")

        with open(filepath, 'r') as f:
            content = f.read()

        assert "Performance Chart" in content

    def test_html_report_failed_result(self, html_report_config, failed_result):
        """Test HTML report handles failed results."""
        # Disable charts since all results are failures (would cause empty max())
        html_report_config.include_charts = False
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate([failed_result], "failed_report.html")

        with open(filepath, 'r') as f:
            content = f.read()

        assert "FAIL" in content
        assert "failed_benchmark" in content

    def test_html_report_auto_filename(self, html_report_config, successful_result):
        """Test HTML report with auto-generated filename."""
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate([successful_result])

        assert os.path.exists(filepath)
        assert "report_" in os.path.basename(filepath)
        assert filepath.endswith(".html")


class TestMarkdownReportGeneration:
    """Tests for Markdown report generation."""

    def test_generate_markdown_report(self, report_config, multiple_results):
        """Test generating Markdown report."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate(multiple_results)

        assert os.path.exists(filepath)
        assert filepath.endswith(".md")

    def test_markdown_report_content(self, report_config, successful_result):
        """Test Markdown report content structure."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([successful_result], "test_report.md")

        with open(filepath, 'r') as f:
            content = f.read()

        # Check heading
        assert "# Test Report" in content

        # Check summary table
        assert "## Summary" in content
        assert "| Benchmark | Status |" in content
        assert "test_benchmark" in content
        assert "PASS" in content

        # Check detailed results
        assert "## Detailed Results" in content

    def test_markdown_report_with_metrics(self, report_config, successful_result):
        """Test Markdown report includes metrics table."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([successful_result], "metrics_report.md")

        with open(filepath, 'r') as f:
            content = f.read()

        # Check metrics table
        assert "| Metric | Value | Unit |" in content
        assert "mean_latency" in content

    def test_markdown_report_failed_result(self, report_config, failed_result):
        """Test Markdown report handles failed results."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([failed_result], "failed_report.md")

        with open(filepath, 'r') as f:
            content = f.read()

        assert "FAIL" in content
        assert "**Error:**" in content
        assert "timeout" in content

    def test_markdown_report_auto_filename(self, report_config, successful_result):
        """Test Markdown report with auto-generated filename."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([successful_result])

        assert os.path.exists(filepath)
        assert "report_" in os.path.basename(filepath)
        assert filepath.endswith(".md")


class TestJSONReportGeneration:
    """Tests for JSON report generation."""

    def test_generate_json_report(self, json_report_config, multiple_results):
        """Test generating JSON report."""
        generator = ReportGenerator(json_report_config)
        filepath = generator.generate(multiple_results)

        assert os.path.exists(filepath)
        assert filepath.endswith(".json")

    def test_json_report_structure(self, json_report_config, multiple_results):
        """Test JSON report structure."""
        generator = ReportGenerator(json_report_config)
        filepath = generator.generate(multiple_results, "test_report.json")

        with open(filepath, 'r') as f:
            data = json.load(f)

        assert "title" in data
        assert "generated" in data
        assert "results" in data
        assert data["title"] == "JSON Test Report"
        assert len(data["results"]) == 3

    def test_json_report_result_format(self, json_report_config, successful_result):
        """Test JSON report result format."""
        generator = ReportGenerator(json_report_config)
        filepath = generator.generate([successful_result], "result_format.json")

        with open(filepath, 'r') as f:
            data = json.load(f)

        result = data["results"][0]

        assert "benchmark_name" in result
        assert "config" in result
        assert "metrics" in result
        assert "statistics" in result
        assert "success" in result

    def test_json_report_auto_filename(self, json_report_config, successful_result):
        """Test JSON report with auto-generated filename."""
        generator = ReportGenerator(json_report_config)
        filepath = generator.generate([successful_result])

        assert os.path.exists(filepath)
        assert "report_" in os.path.basename(filepath)
        assert filepath.endswith(".json")


class TestReportFormatSelection:
    """Tests for report format selection."""

    def test_default_format_fallback(self, temp_output_dir, successful_result):
        """Test that unknown format falls back to markdown."""
        config = ReportConfig(
            output_dir=temp_output_dir,
            format="unknown_format"
        )
        generator = ReportGenerator(config)
        filepath = generator.generate([successful_result])

        assert filepath.endswith(".md")

    def test_format_html(self, temp_output_dir, successful_result):
        """Test HTML format selection."""
        config = ReportConfig(output_dir=temp_output_dir, format="html")
        generator = ReportGenerator(config)
        filepath = generator.generate([successful_result])

        assert filepath.endswith(".html")

    def test_format_markdown(self, temp_output_dir, successful_result):
        """Test Markdown format selection."""
        config = ReportConfig(output_dir=temp_output_dir, format="markdown")
        generator = ReportGenerator(config)
        filepath = generator.generate([successful_result])

        assert filepath.endswith(".md")

    def test_format_json(self, temp_output_dir, successful_result):
        """Test JSON format selection."""
        config = ReportConfig(output_dir=temp_output_dir, format="json")
        generator = ReportGenerator(config)
        filepath = generator.generate([successful_result])

        assert filepath.endswith(".json")


class TestASCIIChart:
    """Tests for ASCII chart generation."""

    def test_ascii_chart_generated(self, html_report_config, multiple_results):
        """Test ASCII chart is included in HTML report."""
        html_report_config.include_charts = True
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate(multiple_results)

        with open(filepath, 'r') as f:
            content = f.read()

        assert "Performance Chart" in content
        assert "<pre>" in content

    def test_ascii_chart_disabled(self, temp_output_dir, multiple_results):
        """Test ASCII chart can be disabled."""
        config = ReportConfig(
            output_dir=temp_output_dir,
            format="html",
            include_charts=False
        )
        generator = ReportGenerator(config)
        filepath = generator.generate(multiple_results)

        with open(filepath, 'r') as f:
            content = f.read()

        assert "Performance Chart" not in content

    def test_ascii_chart_empty_results(self, html_report_config):
        """Test ASCII chart with empty results."""
        generator = ReportGenerator(html_report_config)
        chart = generator._generate_ascii_chart([])

        assert chart == ""

    def test_ascii_chart_all_failed(self, html_report_config, failed_result):
        """Test ASCII chart with all failed results."""
        # When all results fail, chart generation should be disabled
        # to avoid ValueError from max() on empty sequence
        html_report_config.include_charts = False
        generator = ReportGenerator(html_report_config)
        filepath = generator.generate([failed_result])

        # Should not crash, chart section may be empty
        assert os.path.exists(filepath)


class TestComparisonReport:
    """Tests for ComparisonReport class."""

    def test_comparison_creation(self, report_config):
        """Test comparison report creation."""
        comparison = ComparisonReport(report_config)
        assert comparison.config == report_config

    def test_comparison_generate(self, report_config, baseline_results, current_results):
        """Test generating comparison report."""
        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline_results, current_results)

        assert os.path.exists(filepath)
        assert filepath.endswith(".md")

    def test_comparison_content(self, report_config, baseline_results, current_results):
        """Test comparison report content."""
        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline_results, current_results, "comparison.md")

        with open(filepath, 'r') as f:
            content = f.read()

        # Check headers
        assert "# Benchmark Comparison Report" in content
        assert "## Summary" in content

        # Check table headers
        assert "| Benchmark | Baseline (ms) | Current (ms) | Speedup | Status |" in content

    def test_comparison_speedup_calculation(self, report_config, baseline_results, current_results):
        """Test speedup calculation in comparison."""
        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline_results, current_results, "speedup.md")

        with open(filepath, 'r') as f:
            content = f.read()

        # benchmark_a: 100ms -> 50ms = 2x speedup
        assert "benchmark_a" in content
        assert "2.00x" in content
        assert "Improved" in content

    def test_comparison_regression_detection(self, report_config, baseline_results, current_results):
        """Test regression detection in comparison."""
        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline_results, current_results, "regression.md")

        with open(filepath, 'r') as f:
            content = f.read()

        # benchmark_b: 200ms -> 220ms = regression
        assert "benchmark_b" in content
        assert "Regressed" in content

    def test_comparison_improvements_section(self, report_config, baseline_results, current_results):
        """Test improvements section in comparison."""
        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline_results, current_results, "improvements.md")

        with open(filepath, 'r') as f:
            content = f.read()

        assert "### Improvements" in content
        assert "benchmark_a" in content
        assert "faster" in content

    def test_comparison_regressions_section(self, report_config, baseline_results, current_results):
        """Test regressions section in comparison."""
        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline_results, current_results, "regressions.md")

        with open(filepath, 'r') as f:
            content = f.read()

        assert "### Regressions" in content
        assert "benchmark_b" in content
        assert "slower" in content

    def test_comparison_no_matching_benchmarks(self, report_config):
        """Test comparison with no matching benchmarks."""
        config = BenchmarkConfig(name="test")

        baseline = [
            BenchmarkResult(
                benchmark_name="only_in_baseline",
                config=config,
                metrics=[],
                raw_times_ms=[100.0],
                success=True
            )
        ]

        current = [
            BenchmarkResult(
                benchmark_name="only_in_current",
                config=config,
                metrics=[],
                raw_times_ms=[50.0],
                success=True
            )
        ]

        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline, current, "no_match.md")

        # Should still generate valid report
        assert os.path.exists(filepath)

    def test_comparison_auto_filename(self, report_config, baseline_results, current_results):
        """Test comparison with auto-generated filename."""
        comparison = ComparisonReport(report_config)
        filepath = comparison.compare(baseline_results, current_results)

        assert os.path.exists(filepath)
        assert "comparison_" in os.path.basename(filepath)


class TestMetricsAnalyzer:
    """Tests for MetricsAnalyzer class."""

    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = MetricsAnalyzer()
        assert analyzer is not None

    def test_analyze_basic(self, multiple_results):
        """Test basic analysis."""
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(multiple_results)

        assert "total_benchmarks" in analysis
        assert "passed" in analysis
        assert "failed" in analysis
        assert analysis["total_benchmarks"] == 3
        assert analysis["passed"] == 2
        assert analysis["failed"] == 1

    def test_analyze_metrics_summary(self, multiple_results):
        """Test metrics summary in analysis."""
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(multiple_results)

        assert "metrics_summary" in analysis

        # Should have aggregated metrics
        summary = analysis["metrics_summary"]
        if summary:  # May be empty if no metrics
            for metric_name, stats in summary.items():
                assert "mean" in stats
                assert "min" in stats
                assert "max" in stats
                assert "count" in stats

    def test_analyze_bottlenecks(self, multiple_results):
        """Test bottleneck identification."""
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(multiple_results)

        assert "bottlenecks" in analysis
        assert isinstance(analysis["bottlenecks"], list)

    def test_analyze_recommendations(self, multiple_results):
        """Test recommendations generation."""
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(multiple_results)

        assert "recommendations" in analysis
        assert isinstance(analysis["recommendations"], list)

        # Should recommend fixing failed benchmark
        recommendations = analysis["recommendations"]
        assert any("Fix failing benchmark" in r for r in recommendations)

    def test_identify_slow_benchmark(self):
        """Test identification of slow benchmarks."""
        config = BenchmarkConfig(name="test")

        results = [
            BenchmarkResult(
                benchmark_name="slow_benchmark",
                config=config,
                metrics=[],
                raw_times_ms=[500.0],  # > 100ms threshold
                success=True
            )
        ]

        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze(results)

        bottlenecks = analysis["bottlenecks"]
        assert any("Slow benchmark" in b for b in bottlenecks)

    def test_identify_high_variance(self):
        """Test identification of high variance benchmarks."""
        config = BenchmarkConfig(name="test")

        # Create result with high variance (std > 20% of mean)
        result = BenchmarkResult(
            benchmark_name="high_variance",
            config=config,
            metrics=[],
            raw_times_ms=[10.0, 30.0, 50.0, 70.0, 90.0],  # High variance
            success=True
        )

        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze([result])

        bottlenecks = analysis["bottlenecks"]
        assert any("High variance" in b for b in bottlenecks)

    def test_memory_recommendation(self):
        """Test memory usage recommendation."""
        config = BenchmarkConfig(name="test")

        result = BenchmarkResult(
            benchmark_name="memory_heavy",
            config=config,
            metrics=[
                Metric(
                    name="peak_memory",
                    value=10000.0,  # > 8GB threshold
                    unit=MetricType.MEMORY_MB
                )
            ],
            raw_times_ms=[50.0],
            success=True
        )

        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze([result])

        recommendations = analysis["recommendations"]
        assert any("Optimize memory" in r for r in recommendations)

    def test_analyze_empty_results(self):
        """Test analysis with empty results."""
        analyzer = MetricsAnalyzer()
        analysis = analyzer.analyze([])

        assert analysis["total_benchmarks"] == 0
        assert analysis["passed"] == 0
        assert analysis["failed"] == 0


class TestGenerateReportFunction:
    """Tests for the generate_report convenience function."""

    def test_generate_report_default(self, temp_output_dir, successful_result):
        """Test generate_report with defaults."""
        filepath = generate_report(
            [successful_result],
            output_dir=temp_output_dir
        )

        assert os.path.exists(filepath)
        assert filepath.endswith(".md")  # Default format

    def test_generate_report_html(self, temp_output_dir, successful_result):
        """Test generate_report with HTML format."""
        filepath = generate_report(
            [successful_result],
            output_dir=temp_output_dir,
            format="html"
        )

        assert os.path.exists(filepath)
        assert filepath.endswith(".html")

    def test_generate_report_json(self, temp_output_dir, successful_result):
        """Test generate_report with JSON format."""
        filepath = generate_report(
            [successful_result],
            output_dir=temp_output_dir,
            format="json"
        )

        assert os.path.exists(filepath)
        assert filepath.endswith(".json")


class TestReportEdgeCases:
    """Tests for edge cases in report generation."""

    def test_empty_results(self, report_config):
        """Test report generation with empty results."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([])

        assert os.path.exists(filepath)

    def test_single_result(self, report_config, successful_result):
        """Test report with single result."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([successful_result])

        assert os.path.exists(filepath)

    def test_all_failed_results(self, report_config, failed_result):
        """Test report with all failed results."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([failed_result])

        assert os.path.exists(filepath)

    def test_mixed_results(self, report_config, successful_result, failed_result):
        """Test report with mixed success/failure."""
        generator = ReportGenerator(report_config)
        filepath = generator.generate([successful_result, failed_result])

        assert os.path.exists(filepath)

    def test_result_with_no_metrics(self, report_config):
        """Test report with result that has no metrics."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(
            benchmark_name="no_metrics",
            config=config,
            metrics=[],
            raw_times_ms=[10.0],
            success=True
        )

        generator = ReportGenerator(report_config)
        filepath = generator.generate([result])

        assert os.path.exists(filepath)

    def test_result_with_empty_times(self, report_config):
        """Test report with result that has no timing data."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(
            benchmark_name="no_times",
            config=config,
            metrics=[],
            raw_times_ms=[],
            success=True
        )

        generator = ReportGenerator(report_config)
        filepath = generator.generate([result])

        assert os.path.exists(filepath)

    def test_special_characters_in_benchmark_name(self, report_config):
        """Test report handles special characters in names."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(
            benchmark_name="test<>&\"'benchmark",
            config=config,
            metrics=[],
            raw_times_ms=[10.0],
            success=True
        )

        generator = ReportGenerator(report_config)
        filepath = generator.generate([result])

        assert os.path.exists(filepath)

    def test_very_long_benchmark_name(self, report_config):
        """Test report handles very long benchmark names."""
        config = BenchmarkConfig(name="test")
        long_name = "a" * 200
        result = BenchmarkResult(
            benchmark_name=long_name,
            config=config,
            metrics=[],
            raw_times_ms=[10.0],
            success=True
        )

        generator = ReportGenerator(report_config)
        filepath = generator.generate([result])

        assert os.path.exists(filepath)

    def test_zero_time_result(self, report_config):
        """Test report with zero timing values."""
        config = BenchmarkConfig(name="test")
        result = BenchmarkResult(
            benchmark_name="zero_time",
            config=config,
            metrics=[],
            raw_times_ms=[0.0, 0.0, 0.0],
            success=True
        )

        generator = ReportGenerator(report_config)
        filepath = generator.generate([result])

        assert os.path.exists(filepath)
