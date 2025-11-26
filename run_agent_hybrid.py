import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
sys.path.insert(0, str(Path(__file__).parent))
from agent.graph_hybrid import RetailAnalyticsCopilot
console = Console()

def load_questions(filepath: str) -> List[Dict[str, Any]]:
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def save_outputs(outputs: List[Dict[str, Any]], filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')

def display_results(outputs: List[Dict[str, Any]]) -> None:
    table = Table(title="Results Summary")
    table.add_column("ID", style="cyan")
    table.add_column("Answer", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("SQL", style="dim", max_width=40)
    for out in outputs:
        answer_str = str(out["final_answer"])
        if len(answer_str) > 50:
            answer_str = answer_str[:47] + "..."
        sql_str = out.get("sql","")
        if len(sql_str) > 40:
            sql_str = sql_str[:37] + "..."
        table.add_row(out["id"], answer_str, f"{out['confidence']:.2f}", sql_str)
    console.print(table)

@click.command()
@click.option('--batch', required=True, type=click.Path(exists=True))
@click.option('--out', required=True, type=click.Path())
@click.option('--model', default='phi3.5:3.8b-mini-instruct-q4_K_M')
@click.option('--verbose', '-v', is_flag=True)
def main(batch: str, out: str, model: str, verbose: bool):
    console.print("[bold blue]Retail Analytics Copilot[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Input: {batch}")
    console.print(f"Output: {out}")
    console.print()
    console.print("[dim]Loading questions...[/dim]")
    questions = load_questions(batch)
    console.print(f"Loaded {len(questions)} questions")
    console.print("[dim]Initializing agent...[/dim]")
    try:
        agent = RetailAnalyticsCopilot(model_name=model)
        console.print("[green]Agent initialized[/green]")
    except Exception as e:
        console.print(f"[red]Failed to initialize agent: {e}[/red]")
        console.print("[yellow]Make sure Ollama is running with the model pulled:[/yellow]")
        console.print(f"  ollama pull {model}")
        sys.exit(1)
    outputs = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Processing...", total=len(questions))
        for q in questions:
            progress.update(task, description=f"Processing: {q['id']}")
            try:
                output, trace = agent.run(question=q["question"], format_hint=q["format_hint"], question_id=q["id"])
                outputs.append(output)
                if verbose:
                    console.print(f"\n[cyan]{q['id']}[/cyan]")
                    console.print(f"  Question: {q['question'][:80]}...")
                    console.print(f"  Answer: {output['final_answer']}")
                    console.print(f"  Confidence: {output['confidence']}")
                    if output['sql']:
                        console.print(f"  SQL: {output['sql'][:100]}...")
                    console.print(f"  Citations: {output['citations']}")
            except Exception as e:
                console.print(f"[red]Error on {q['id']}: {e}[/red]")
                outputs.append({"id": q["id"], "final_answer": None, "sql": "", "confidence": 0.0, "explanation": f"Error: {str(e)}", "citations": []})
            progress.advance(task)
    console.print(f"\n[dim]Saving to {out}...[/dim]")
    save_outputs(outputs, out)
    console.print(f"[green]Saved {len(outputs)} results[/green]")
    console.print()
    display_results(outputs)
    successful = sum(1 for o in outputs if o["final_answer"] is not None)
    console.print(f"\n[bold]Summary:[/bold] {successful}/{len(outputs)} successful")

if __name__ == "__main__":
    main()
