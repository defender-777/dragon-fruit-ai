import os
import time
import logging
from io import StringIO
from unittest.mock import MagicMock
import app.model

# Mock machine learning weights so backend tests load instantly
app.model.load_model = MagicMock()

from fastapi.testclient import TestClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from app.main import app, logger

console = Console()
client = TestClient(app)

def test_logs():
    # Setup our intercepting Stream Handler so we can assert on log emissions
    log_stream = StringIO()
    stream_handler = logging.StreamHandler(log_stream)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)
    
    console.print("\n[dim]Firing request: GET / (Generating foundational tracing behavior)[/dim]")
    client.get("/")
    
    console.print("[dim]Firing request: POST /predict (Injecting bad configuration to trace exceptions)[/dim]")
    client.post("/predict", files={"file": ("test.jpg", b"corrupted data", "image/jpeg")})
    
    logs_output = log_stream.getvalue()
    logger.removeHandler(stream_handler)
    
    table = Table(title="Logging Verification Matrix", show_header=True, header_style="bold magenta")
    table.add_column("Log Event Objective", style="cyan", width=35)
    table.add_column("Presence Validation", justify="center")

    has_incoming = "Incoming request: GET /" in logs_output
    has_completed = "Completed request: GET /" in logs_output
    has_app_error = "HTTPException: Invalid image file" in logs_output
    has_400_status = "Status: 400" in logs_output
    
    table.add_row("Middleware Incoming Request trace", "[green]Verified[/green]" if has_incoming else "[red]Missing[/red]")
    table.add_row("Middleware Success Response trace", "[green]Verified[/green]" if has_completed else "[red]Missing[/red]")
    table.add_row("Structured Endpoint Error Log", "[green]Verified[/green]" if has_app_error else "[red]Missing[/red]")
    table.add_row("Failure Tracked in Response Output", "[green]Verified[/green]" if has_400_status else "[red]Missing[/red]")

    console.print(table)
    
    console.print("\n[bold yellow]Raw Harvested Logs Output Simulation:[/bold yellow]")
    console.print(Panel(logs_output.strip() or "No logs produced", border_style="yellow"))

    assert has_incoming, "Incoming request trace logs missing"
    assert has_completed, "Completion request trace logs missing"
    assert has_app_error, "Structured endpoint errors missing in log emissions"
    assert has_400_status, "API failure 400 code was unaccounted for in response tracing logs"

if __name__ == "__main__":
    console.print(Panel.fit("[bold blue]Structured Logging & Tracing Test Suite[/bold blue]", border_style="cyan"))
    try:
        test_logs()
        console.print(Panel.fit("[bold green]ALL LOGGING SYSTEMS VERIFIED[/bold green]", border_style="green"))
    except AssertionError as e:
        console.print(Panel.fit(f"[bold red]TEST FAILED:[/bold red] {e}", border_style="red"))
