import os
import time
from unittest.mock import MagicMock
import app.model

# Mock before importing the app
app.model.load_model = MagicMock()

from fastapi.testclient import TestClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from app.main import app

console = Console()
client = TestClient(app)

def test_home_rate_limit():
    table = Table(title="Rate Limit Test: GET / (5 requests/minute)", show_header=True, header_style="bold magenta")
    table.add_column("Request #", style="dim", width=12)
    table.add_column("Status Code", justify="right")
    table.add_column("Result", justify="left")

    success_count = 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Testing home endpoint...", total=6)
        for i in range(6):
            response = client.get("/")
            if response.status_code == 200:
                success_count += 1
                table.add_row(str(i+1), "200", "[green]Success[/green]")
            elif response.status_code == 429:
                table.add_row(str(i+1), "429", "[blue]Rate Limited[/blue]")
            else:
                table.add_row(str(i+1), str(response.status_code), "[red]Unexpected[/red]")
            progress.advance(task)
            time.sleep(0.1)

    console.print(table)
    assert success_count == 5, f"Should succeed exactly 5 times, but got {success_count}"
    console.print("[bold green]Home rate limit functional.[/bold green]\n")

def test_predict_auth():
    table = Table(title="Authentication Test: POST /predict", show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan", width=25)
    table.add_column("Status Code", justify="right")
    table.add_column("Result", justify="left")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Testing authentication...", total=2)
        
        # Missing API Key
        res_missing = client.post("/predict")
        if res_missing.status_code == 403:
            table.add_row("Missing API Key", "403", "[green]Caught[/green]")
        else:
            table.add_row("Missing API Key", str(res_missing.status_code), "[red]Failed[/red]")
        progress.advance(task)
        time.sleep(0.2)
        
        # Invalid API Key
        res_invalid = client.post("/predict", headers={"X-API-Key": "wrong-key"})
        if res_invalid.status_code == 403:
            table.add_row("Invalid API Key", "403", "[green]Caught[/green]")
        else:
            table.add_row("Invalid API Key", str(res_invalid.status_code), "[red]Failed[/red]")
        progress.advance(task)
        time.sleep(0.2)

    console.print(table)
    assert res_missing.status_code == 403, "Expected 403 Forbidden without API Key"
    assert res_invalid.status_code == 403, "Expected 403 Forbidden with invalid API Key"
    console.print("[bold green]API Key validation functional.[/bold green]\n")

def test_predict_rate_limit():
    table = Table(title="Rate Limit Test: POST /predict (10 requests/minute)", show_header=True, header_style="bold magenta")
    table.add_column("Request #", style="dim", width=12)
    table.add_column("Status Code", justify="right")
    table.add_column("Result", justify="left")

    headers = {"X-API-Key": os.getenv("API_KEY", "default-secret-key")}
    success_count = 0
    limit_hit = False
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Testing predict endpoint...", total=11)
        for i in range(11):
            res = client.post(
                "/predict", 
                headers=headers, 
                files={"file": ("test.jpg", b"not an image", "image/jpeg")}
            )
            if res.status_code != 429:
                success_count += 1
                table.add_row(str(i+1), str(res.status_code), "[cyan]Allowed[/cyan]")
            elif res.status_code == 429:
                limit_hit = True
                table.add_row(str(i+1), "429", "[blue]Rate Limited[/blue]")
            
            progress.advance(task)
            time.sleep(0.1)

    console.print(table)
    assert success_count == 10, f"Should process exactly 10 requests, got {success_count}"
    assert limit_hit, "Should be rate limited on 11th request"
    console.print("[bold green]Predict rate limit functional.[/bold green]\n")

if __name__ == "__main__":
    console.print(Panel.fit("[bold blue]Rate Limiter & Authentication Test Suite[/bold blue]", border_style="cyan"))
    try:
        test_home_rate_limit()
        test_predict_auth()
        test_predict_rate_limit()
        console.print(Panel.fit("[bold green]ALL TESTS PASSED SUCCESSFULLY[/bold green]", border_style="green"))
    except AssertionError as e:
        console.print(Panel.fit(f"[bold red]TEST FAILED:[/bold red] {e}", border_style="red"))
