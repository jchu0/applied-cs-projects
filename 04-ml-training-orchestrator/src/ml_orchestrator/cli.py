"""Command-line interface for ML Training Orchestrator."""

import asyncio
import json
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """ML Training Orchestrator CLI."""
    pass


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--log-level", default="info", help="Log level")
def serve(host: str, port: int, reload: bool, log_level: str):
    """Start the API server."""
    import uvicorn

    from ml_orchestrator.utils.logging import setup_logging

    setup_logging(level=log_level.upper())

    uvicorn.run(
        "ml_orchestrator.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )


@main.group()
def jobs():
    """Job management commands."""
    pass


@jobs.command("list")
@click.option("--status", help="Filter by status")
@click.option("--limit", default=20, help="Number of jobs to show")
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def list_jobs(status: Optional[str], limit: int, api_url: str):
    """List training jobs."""
    import httpx

    params = {"limit": limit}
    if status:
        params["status"] = status

    with httpx.Client() as client:
        response = client.get(f"{api_url}/api/v1/jobs", params=params)
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

        jobs = response.json()

    table = Table(title="Training Jobs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Priority")
    table.add_column("Progress")
    table.add_column("Created")

    for job in jobs:
        status_color = {
            "running": "green",
            "completed": "blue",
            "failed": "red",
            "queued": "yellow",
        }.get(job["status"], "white")

        table.add_row(
            job["id"][:8] + "...",
            job["name"][:30],
            f"[{status_color}]{job['status']}[/{status_color}]",
            str(job["priority"]),
            f"{job['progress_percent']:.1f}%",
            job["created_at"][:19],
        )

    console.print(table)


@jobs.command("get")
@click.argument("job_id")
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def get_job(job_id: str, api_url: str):
    """Get job details."""
    import httpx

    with httpx.Client() as client:
        response = client.get(f"{api_url}/api/v1/jobs/{job_id}/details")
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

        job = response.json()

    console.print_json(json.dumps(job, indent=2))


@jobs.command("submit")
@click.option("--name", required=True, help="Job name")
@click.option("--script", required=True, help="Training script path")
@click.option("--user-id", required=True, help="User ID")
@click.option("--cpus", default=1, help="Number of CPUs")
@click.option("--memory", default=4.0, help="Memory in GB")
@click.option("--gpus", default=0, help="Number of GPUs")
@click.option("--priority", default="normal", help="Priority level")
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def submit_job(
    name: str,
    script: str,
    user_id: str,
    cpus: int,
    memory: float,
    gpus: int,
    priority: str,
    api_url: str,
):
    """Submit a new training job."""
    import httpx

    priority_map = {
        "lowest": 0,
        "low": 25,
        "normal": 50,
        "high": 75,
        "highest": 100,
    }

    payload = {
        "name": name,
        "user_id": user_id,
        "config": {"script_path": script},
        "resources": {
            "cpus": cpus,
            "memory_gb": memory,
            "gpus": gpus,
        },
        "priority": priority_map.get(priority.lower(), 50),
    }

    with httpx.Client() as client:
        response = client.post(f"{api_url}/api/v1/jobs", json=payload)
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

        job = response.json()

    console.print(f"[green]Job submitted successfully![/green]")
    console.print(f"Job ID: {job['id']}")


@jobs.command("cancel")
@click.argument("job_id")
@click.option("--reason", default="User cancelled", help="Cancellation reason")
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def cancel_job(job_id: str, reason: str, api_url: str):
    """Cancel a job."""
    import httpx

    with httpx.Client() as client:
        response = client.post(
            f"{api_url}/api/v1/jobs/{job_id}/cancel",
            params={"reason": reason},
        )
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

    console.print(f"[yellow]Job {job_id} cancelled[/yellow]")


@main.group()
def workers():
    """Worker management commands."""
    pass


@workers.command("list")
@click.option("--healthy-only", is_flag=True, help="Show only healthy workers")
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def list_workers(healthy_only: bool, api_url: str):
    """List workers."""
    import httpx

    params = {"healthy_only": healthy_only}

    with httpx.Client() as client:
        response = client.get(f"{api_url}/api/v1/resources/workers", params=params)
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

        workers = response.json()

    table = Table(title="Workers")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Hostname")
    table.add_column("Status")
    table.add_column("CPUs")
    table.add_column("Memory")
    table.add_column("GPUs")
    table.add_column("Jobs")

    for worker in workers:
        status_color = {
            "ready": "green",
            "busy": "yellow",
            "unhealthy": "red",
        }.get(worker["status"], "white")

        table.add_row(
            worker["id"][:8] + "...",
            worker["hostname"],
            f"[{status_color}]{worker['status']}[/{status_color}]",
            f"{worker['allocated_cpus']}/{worker['cpus']}",
            f"{worker['allocated_memory_gb']:.1f}/{worker['memory_gb']:.1f}",
            f"{worker['allocated_gpus']}/{worker['gpus']}",
            str(len(worker["current_jobs"])),
        )

    console.print(table)


@workers.command("register")
@click.option("--hostname", required=True, help="Worker hostname")
@click.option("--ip", required=True, help="Worker IP address")
@click.option("--cpus", required=True, type=int, help="Number of CPUs")
@click.option("--memory", required=True, type=float, help="Memory in GB")
@click.option("--gpus", default=0, type=int, help="Number of GPUs")
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def register_worker(
    hostname: str, ip: str, cpus: int, memory: float, gpus: int, api_url: str
):
    """Register a new worker."""
    import httpx

    payload = {
        "hostname": hostname,
        "ip_address": ip,
        "cpus": cpus,
        "memory_gb": memory,
        "gpus": gpus,
    }

    with httpx.Client() as client:
        response = client.post(f"{api_url}/api/v1/resources/workers", json=payload)
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

        worker = response.json()

    console.print(f"[green]Worker registered successfully![/green]")
    console.print(f"Worker ID: {worker['id']}")


@main.group()
def experiments():
    """Experiment tracking commands."""
    pass


@experiments.command("list")
@click.option("--user-id", help="Filter by user ID")
@click.option("--limit", default=20, help="Number to show")
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def list_experiments(user_id: Optional[str], limit: int, api_url: str):
    """List experiments."""
    import httpx

    params = {"limit": limit}
    if user_id:
        params["user_id"] = user_id

    with httpx.Client() as client:
        response = client.get(f"{api_url}/api/v1/experiments", params=params)
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

        experiments = response.json()

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Runs")
    table.add_column("Best Metric")
    table.add_column("Created")

    for exp in experiments:
        table.add_row(
            exp["id"][:8] + "...",
            exp["name"][:30],
            exp["status"],
            str(len(exp["job_ids"])),
            f"{exp['best_metric_value']:.4f}" if exp["best_metric_value"] else "-",
            exp["created_at"][:19],
        )

    console.print(table)


@main.command()
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def status(api_url: str):
    """Show system status."""
    import httpx

    with httpx.Client() as client:
        response = client.get(f"{api_url}/health/stats")
        if response.status_code != 200:
            console.print(f"[red]Error: {response.text}[/red]")
            return

        stats = response.json()

    console.print("[bold]System Status[/bold]")
    console.print()

    # Scheduler
    scheduler = stats.get("scheduler", {})
    console.print(f"[cyan]Scheduler:[/cyan]")
    console.print(f"  Running: {scheduler.get('running', False)}")
    console.print(f"  Queue Size: {scheduler.get('queue', {}).get('size', 0)}")
    console.print()

    # Jobs
    job_mgr = stats.get("job_manager", {})
    console.print(f"[cyan]Jobs:[/cyan]")
    console.print(f"  Total: {job_mgr.get('total_jobs', 0)}")
    jobs_by_status = job_mgr.get("jobs_by_status", {})
    for status, count in jobs_by_status.items():
        if count > 0:
            console.print(f"  {status}: {count}")
    console.print()

    # GPU
    gpu = stats.get("gpu_manager", {})
    console.print(f"[cyan]GPUs:[/cyan]")
    console.print(f"  Total: {gpu.get('total_gpus', 0)}")
    console.print(f"  Available: {gpu.get('available_gpus', 0)}")


if __name__ == "__main__":
    main()
