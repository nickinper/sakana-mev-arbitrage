#!/usr/bin/env python3
"""
Terminal-based Training UI using Rich
Beautiful console interface for training visualization
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
import json

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box
from rich.syntax import Syntax
from rich.chart import AsciiChart

console = Console()


class TerminalTrainingUI:
    """Rich terminal interface for training visualization"""
    
    def __init__(self):
        self.console = console
        self.layout = self._create_layout()
        self.training_data = {
            'generation': 0,
            'best_fitness': 0.0,
            'avg_fitness': 0.0,
            'total_profit': 0.0,
            'opportunities_found': 0,
            'agents': [],
            'events': [],
            'fitness_history': [],
            'profit_history': []
        }
        self.is_paused = False
        self.start_time = datetime.now()
        
    def _create_layout(self) -> Layout:
        """Create the layout structure"""
        layout = Layout()
        
        # Split into header, body, and footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Split body into left and right
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left into top and bottom
        layout["left"].split_column(
            Layout(name="stats", size=10),
            Layout(name="agents"),
            Layout(name="charts", size=12)
        )
        
        # Right panel for events
        layout["right"].split_column(
            Layout(name="events"),
            Layout(name="controls", size=8)
        )
        
        return layout
    
    def _make_header(self) -> Panel:
        """Create header panel"""
        elapsed = datetime.now() - self.start_time
        status = "[red]PAUSED[/red]" if self.is_paused else "[green]RUNNING[/green]"
        
        header_text = Text.from_markup(
            f"[bold blue]🌊 Sakana MEV Training System[/bold blue] | "
            f"Status: {status} | "
            f"Elapsed: {elapsed.seconds // 60}m {elapsed.seconds % 60}s"
        )
        
        return Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE
        )
    
    def _make_stats_panel(self) -> Panel:
        """Create statistics panel"""
        stats_table = Table(show_header=False, box=None, expand=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bright_green")
        
        stats_table.add_row("Generation", f"{self.training_data['generation']}")
        stats_table.add_row("Best Fitness", f"{self.training_data['best_fitness']:.4f}")
        stats_table.add_row("Avg Fitness", f"{self.training_data['avg_fitness']:.4f}")
        stats_table.add_row("Total Profit", f"${self.training_data['total_profit']:.2f}")
        stats_table.add_row("Opportunities", f"{self.training_data['opportunities_found']}")
        
        return Panel(
            stats_table,
            title="[bold]Training Statistics[/bold]",
            border_style="green",
            box=box.ROUNDED
        )
    
    def _make_agents_table(self) -> Panel:
        """Create agents table"""
        table = Table(
            title="Current Population",
            box=box.SIMPLE_HEAD,
            show_lines=True
        )
        
        table.add_column("Agent ID", style="cyan", width=20)
        table.add_column("Fitness", style="green", width=10)
        table.add_column("Real Profit", style="yellow", width=12)
        table.add_column("Success Rate", style="magenta", width=12)
        table.add_column("Strategy", style="blue", width=20)
        
        # Sort agents by fitness
        agents = sorted(
            self.training_data['agents'], 
            key=lambda x: x.get('fitness', 0), 
            reverse=True
        )[:10]  # Top 10
        
        for agent in agents:
            strategy = f"${agent.get('min_profit', 0)}/g{agent.get('gas_mult', 0):.1f}"
            table.add_row(
                agent.get('id', 'Unknown'),
                f"{agent.get('fitness', 0):.4f}",
                f"${agent.get('real_profit', 0):.2f}",
                f"{agent.get('success_rate', 0):.1%}",
                strategy
            )
        
        return Panel(table, border_style="blue", box=box.ROUNDED)
    
    def _make_charts(self) -> Panel:
        """Create ASCII charts for fitness and profit"""
        # Fitness chart
        if len(self.training_data['fitness_history']) > 1:
            fitness_values = self.training_data['fitness_history'][-20:]  # Last 20
            fitness_chart = self._create_ascii_chart(
                fitness_values, 
                "Fitness Evolution",
                height=5
            )
        else:
            fitness_chart = "[dim]No fitness data yet...[/dim]"
        
        # Profit chart
        if len(self.training_data['profit_history']) > 1:
            profit_values = self.training_data['profit_history'][-20:]  # Last 20
            profit_chart = self._create_ascii_chart(
                profit_values,
                "Profit Trend",
                height=5
            )
        else:
            profit_chart = "[dim]No profit data yet...[/dim]"
        
        combined = f"{fitness_chart}\n\n{profit_chart}"
        
        return Panel(
            combined,
            title="[bold]Performance Charts[/bold]",
            border_style="yellow",
            box=box.ROUNDED
        )
    
    def _create_ascii_chart(self, values: List[float], title: str, height: int = 5) -> str:
        """Create simple ASCII chart"""
        if not values:
            return ""
        
        max_val = max(values) if max(values) > 0 else 1
        min_val = min(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        chart_lines = []
        chart_lines.append(f"[bold]{title}[/bold]")
        chart_lines.append(f"Max: {max_val:.2f}")
        
        # Create bars
        for i in range(height, 0, -1):
            line = ""
            threshold = min_val + (i / height) * range_val
            
            for val in values:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            
            chart_lines.append(line)
        
        chart_lines.append("─" * len(values))
        
        return "\n".join(chart_lines)
    
    def _make_events_panel(self) -> Panel:
        """Create events log panel"""
        events = self.training_data['events'][-15:]  # Last 15 events
        
        event_lines = []
        for event in events:
            timestamp = event.get('time', '')
            message = event.get('message', '')
            event_type = event.get('type', 'info')
            
            if event_type == 'profit':
                color = "green"
            elif event_type == 'loss':
                color = "red"
            else:
                color = "blue"
            
            event_lines.append(f"[{color}]{timestamp}[/{color}] {message}")
        
        events_text = "\n".join(event_lines) if event_lines else "[dim]No events yet...[/dim]"
        
        return Panel(
            events_text,
            title="[bold]Event Log[/bold]",
            border_style="cyan",
            box=box.ROUNDED
        )
    
    def _make_controls_panel(self) -> Panel:
        """Create controls panel"""
        controls_text = """[bold]Controls:[/bold]
        
[yellow]P[/yellow] - Pause/Resume
[yellow]S[/yellow] - Save Checkpoint
[yellow]E[/yellow] - Export Results
[yellow]Q[/yellow] - Quit Training

[dim]Status updates every second[/dim]"""
        
        return Panel(
            controls_text,
            border_style="magenta",
            box=box.ROUNDED
        )
    
    def _make_footer(self) -> Panel:
        """Create footer panel"""
        footer_text = Text.from_markup(
            "[dim]Press [bold]Q[/bold] to quit | [bold]P[/bold] to pause | "
            "[bold]S[/bold] to save | [bold]E[/bold] to export[/dim]"
        )
        
        return Panel(
            Align.center(footer_text),
            style="bright_black",
            box=box.HORIZONTALS
        )
    
    def update_display(self):
        """Update all display panels"""
        self.layout["header"].update(self._make_header())
        self.layout["stats"].update(self._make_stats_panel())
        self.layout["agents"].update(self._make_agents_table())
        self.layout["charts"].update(self._make_charts())
        self.layout["events"].update(self._make_events_panel())
        self.layout["controls"].update(self._make_controls_panel())
        self.layout["footer"].update(self._make_footer())
    
    def update_generation(self, generation: int, best_fitness: float, avg_fitness: float):
        """Update generation data"""
        self.training_data['generation'] = generation
        self.training_data['best_fitness'] = best_fitness
        self.training_data['avg_fitness'] = avg_fitness
        self.training_data['fitness_history'].append(best_fitness)
        
        self.add_event(f"Generation {generation} completed", "info")
    
    def update_agents(self, agents: List[Dict]):
        """Update agent population"""
        self.training_data['agents'] = agents
    
    def add_opportunity(self, opportunity: Dict):
        """Add found opportunity"""
        self.training_data['opportunities_found'] += 1
        profit = opportunity.get('net_profit_usd', 0)
        self.add_event(f"Found opportunity: ${profit:.2f}", "profit")
    
    def add_profit(self, amount: float, success: bool):
        """Add profit/loss"""
        self.training_data['total_profit'] += amount
        self.training_data['profit_history'].append(self.training_data['total_profit'])
        
        event_type = "profit" if amount > 0 else "loss"
        self.add_event(f"Trade {'succeeded' if success else 'failed'}: ${amount:.2f}", event_type)
    
    def add_event(self, message: str, event_type: str = "info"):
        """Add event to log"""
        self.training_data['events'].append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'message': message,
            'type': event_type
        })
        
        # Keep only last 50 events
        if len(self.training_data['events']) > 50:
            self.training_data['events'] = self.training_data['events'][-50:]
    
    def pause(self):
        """Pause training"""
        self.is_paused = True
        self.add_event("Training paused", "info")
    
    def resume(self):
        """Resume training"""
        self.is_paused = False
        self.add_event("Training resumed", "info")
    
    def save_checkpoint(self):
        """Save checkpoint"""
        filename = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        self.add_event(f"Checkpoint saved: {filename}", "info")
    
    def export_results(self):
        """Export results"""
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        self.add_event(f"Results exported: {filename}", "info")
    
    async def run(self, update_callback=None):
        """Run the terminal UI"""
        with Live(
            self.layout,
            console=self.console,
            screen=True,
            refresh_per_second=2
        ) as live:
            try:
                while True:
                    # Update display
                    self.update_display()
                    
                    # Call update callback if provided
                    if update_callback and not self.is_paused:
                        await update_callback(self)
                    
                    # Check for keyboard input (non-blocking)
                    # Note: In real implementation, would use keyboard library
                    await asyncio.sleep(0.5)
                    
            except KeyboardInterrupt:
                self.add_event("Training stopped by user", "info")
                self.save_checkpoint()


# Demo function to test the UI
async def demo_training(ui: TerminalTrainingUI):
    """Demo training loop"""
    import random
    
    generation = 0
    agents = [
        {
            'id': f'agent_{i}',
            'fitness': random.random(),
            'real_profit': random.uniform(0, 1000),
            'success_rate': random.random(),
            'min_profit': random.choice([50, 100, 150]),
            'gas_mult': random.uniform(1.2, 2.0)
        }
        for i in range(10)
    ]
    
    while True:
        if not ui.is_paused:
            generation += 1
            
            # Update generation
            best_fitness = max(a['fitness'] for a in agents)
            avg_fitness = sum(a['fitness'] for a in agents) / len(agents)
            ui.update_generation(generation, best_fitness, avg_fitness)
            
            # Update agents
            for agent in agents:
                agent['fitness'] += random.uniform(-0.1, 0.1)
                agent['fitness'] = max(0, min(1, agent['fitness']))
            ui.update_agents(agents)
            
            # Randomly add opportunities
            if random.random() > 0.7:
                ui.add_opportunity({
                    'net_profit_usd': random.uniform(50, 200)
                })
            
            # Randomly add profits
            if random.random() > 0.8:
                profit = random.uniform(-50, 150)
                ui.add_profit(profit, profit > 0)
        
        await asyncio.sleep(2)


if __name__ == "__main__":
    # Test the terminal UI
    ui = TerminalTrainingUI()
    
    console.print("[bold green]Starting Terminal Training UI Demo...[/bold green]")
    console.print("[dim]This is a demo showing the UI capabilities[/dim]\n")
    
    asyncio.run(ui.run(demo_training))