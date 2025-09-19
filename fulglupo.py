"""
Multi-Agent Warehouse Simulation with Task Allocation

A 2D grid simulation where multiple agents navigate to complete delivery tasks
while avoiding collisions. Uses A* path planning with reservation tables.

Features:
- Multiple agents with different colors and tasks
- Central task allocation (greedy nearest or auction-based)
- Collision avoidance using reservation tables
- Real-time visualization with agent paths and task status
- Performance metrics tracking

How to run:
1. Execute the script: python multi_agent_simulation.py
2. Use GUI controls to adjust parameters and start simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
import heapq
import math
import time
import random
from enum import Enum
from typing import List, Tuple, Set, Dict, Optional, Deque
from collections import deque, defaultdict
import matplotlib.patches as patches


# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_agent_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MultiAgentSimulation")


class TaskStatus(Enum):
    PENDING = 1
    ASSIGNED = 2
    IN_PROGRESS = 3
    COMPLETED = 4


class AllocationStrategy(Enum):
    GREEDY_NEAREST = 1
    AUCTION = 2


class Task:
    """Represents a delivery task with package location and destination."""
    
    def __init__(self, task_id: int, package_location: Tuple[int, int], 
                 delivery_location: Tuple[int, int]):
        self.task_id = task_id
        self.package_location = package_location
        self.delivery_location = delivery_location
        self.status = TaskStatus.PENDING
        self.assigned_agent = None
        self.creation_time = time.time()
        self.completion_time = None
    
    def __str__(self):
        return f"Task {self.task_id}: {self.package_location} -> {self.delivery_location} ({self.status.name})"


class Agent:
    """Represents an autonomous agent that can complete tasks."""
    
    def __init__(self, agent_id: int, start_position: Tuple[int, int], color: str):
        self.agent_id = agent_id
        self.position = start_position
        self.color = color
        self.current_task = None
        self.path = []
        self.current_path_index = 0
        self.reserved_cells = {}  # cell -> timestep
        self.waiting_time = 0
        self.tasks_completed = 0
        self.total_distance = 0
        self.status = "IDLE"
    
    def assign_task(self, task: Task):
        """Assign a task to this agent."""
        self.current_task = task
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = self
        self.status = "MOVING_TO_PICKUP"
    
    def clear_task(self):
        """Clear the current task."""
        if self.current_task:
            self.current_task.completion_time = time.time()
            self.current_task.status = TaskStatus.COMPLETED
            self.tasks_completed += 1
            self.current_task = None
        self.status = "IDLE"
        self.path = []
        self.reserved_cells = {}


class GridMap:
    """Represents a 2D grid map with obstacles."""
    
    def __init__(self, width: int, height: int, allow_diagonal: bool = False):
        self.width = width
        self.height = height
        self.allow_diagonal = allow_diagonal
        self.obstacles = np.zeros((height, width), dtype=bool)
        self.reservation_table = defaultdict(dict)  # timestep -> cell -> agent_id
    
    def set_obstacle(self, x: int, y: int, value: bool = True):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles[y, x] = value
    
    def is_obstacle(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.obstacles[y, x]
        return True  # Treat out-of-bound as obstacle
    
    def is_cell_reserved(self, x: int, y: int, timestep: int) -> bool:
        """Check if a cell is reserved at a given timestep."""
        return timestep in self.reservation_table and (x, y) in self.reservation_table[timestep]
    
    def reserve_cell(self, x: int, y: int, timestep: int, agent_id: int):
        """Reserve a cell for an agent at a specific timestep."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.reservation_table[timestep][(x, y)] = agent_id
    
    def clear_reservation(self, x: int, y: int, timestep: int):
        """Clear reservation for a cell at a specific timestep."""
        if timestep in self.reservation_table and (x, y) in self.reservation_table[timestep]:
            del self.reservation_table[timestep][(x, y)]
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connected
        
        if self.allow_diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # 8-connected
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.is_obstacle(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def find_valid_position(self) -> Optional[Tuple[int, int]]:
        """Find a valid position that's not an obstacle."""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_obstacle(x, y):
                    valid_positions.append((x, y))
        
        if valid_positions:
            return valid_positions[np.random.randint(0, len(valid_positions))]
        return None
    
    def generate_random_obstacles(self, density: float = 0.2, cluster_size: int = 1):
        """Generate random obstacles, optionally in clusters."""
        if cluster_size <= 1:
            # Simple random obstacles
            self.obstacles = np.random.random((self.height, self.width)) < density
        else:
            # Generate obstacle clusters
            self.obstacles = np.zeros((self.height, self.width), dtype=bool)
            num_clusters = int(density * self.width * self.height / (cluster_size * cluster_size))
            
            for _ in range(num_clusters):
                cx = np.random.randint(0, self.width - cluster_size + 1)
                cy = np.random.randint(0, self.height - cluster_size + 1)
                
                for dx in range(cluster_size):
                    for dy in range(cluster_size):
                        if np.random.random() < 0.7:  # 70% chance to fill each cell in cluster
                            self.set_obstacle(cx + dx, cy + dy, True)


class AStarPlanner:
    """A* path planning algorithm implementation."""
    
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan a path from start to goal using A* algorithm."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        open_set_hash = {start}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            for neighbor in self.grid_map.get_neighbors(current[0], current[1]):
                tentative_g_score = g_score.get(current, float('inf')) + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return []  # No path found


class SwarmSimulator:
    """Central simulator that manages agents, tasks, and allocation."""
    
    def __init__(self, grid_map: GridMap, num_agents: int = 5, 
                 allocation_strategy: AllocationStrategy = AllocationStrategy.GREEDY_NEAREST,
                 replanning_frequency: int = 5):
        self.grid_map = grid_map
        self.num_agents = num_agents
        self.allocation_strategy = allocation_strategy
        self.replanning_frequency = replanning_frequency
        self.agents = []
        self.tasks = []
        self.completed_tasks = []
        self.current_time = 0
        self.planner = AStarPlanner(grid_map)
        self.agent_colors = ['red', 'blue', 'green', 'orange', 'purple', 
                            'cyan', 'magenta', 'yellow', 'brown', 'pink']
        
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize agents with random positions."""
        for i in range(self.num_agents):
            pos = self.grid_map.find_valid_position()
            if pos:
                color = self.agent_colors[i % len(self.agent_colors)]
                self.agents.append(Agent(i, pos, color))
    
    def generate_task(self):
        """Generate a new random task."""
        package_loc = self.grid_map.find_valid_position()
        delivery_loc = self.grid_map.find_valid_position()
        
        if package_loc and delivery_loc and package_loc != delivery_loc:
            task_id = len(self.tasks) + len(self.completed_tasks)
            task = Task(task_id, package_loc, delivery_loc)
            self.tasks.append(task)
            return task
        return None
    
    def allocate_tasks_greedy(self):
        """Allocate tasks to agents using greedy nearest approach."""
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        idle_agents = [a for a in self.agents if a.status == "IDLE"]
        
        for task in pending_tasks:
            if not idle_agents:
                break
                
            # Find the closest idle agent
            min_dist = float('inf')
            best_agent = None
            
            for agent in idle_agents:
                dist = self.planner.heuristic(agent.position, task.package_location)
                if dist < min_dist:
                    min_dist = dist
                    best_agent = agent
            
            if best_agent:
                best_agent.assign_task(task)
                idle_agents.remove(best_agent)
                logger.info(f"Assigned task {task.task_id} to agent {best_agent.agent_id}")
    
    def allocate_tasks_auction(self):
        """Allocate tasks to agents using auction-based approach."""
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        idle_agents = [a for a in self.agents if a.status == "IDLE"]
        
        for task in pending_tasks:
            if not idle_agents:
                break
                
            # Run auction for this task
            bids = []
            for agent in idle_agents:
                # Bid is based on distance to task (lower is better)
                bid = self.planner.heuristic(agent.position, task.package_location)
                bids.append((bid, agent))
            
            if bids:
                # Select agent with lowest bid (closest distance)
                bids.sort(key=lambda x: x[0])
                best_bid, best_agent = bids[0]
                best_agent.assign_task(task)
                idle_agents.remove(best_agent)
                logger.info(f"Auction: Assigned task {task.task_id} to agent {best_agent.agent_id} with bid {best_bid}")
    
    def allocate_tasks(self):
        """Allocate tasks based on the selected strategy."""
        if self.allocation_strategy == AllocationStrategy.GREEDY_NEAREST:
            self.allocate_tasks_greedy()
        else:
            self.allocate_tasks_auction()
    
    def plan_path_for_agent(self, agent: Agent):
        """Plan a path for an agent based on its current task."""
        if not agent.current_task:
            return
        
        if agent.status == "MOVING_TO_PICKUP":
            goal = agent.current_task.package_location
        elif agent.status == "MOVING_TO_DELIVERY":
            goal = agent.current_task.delivery_location
        else:
            return
        
        path = self.planner.plan(agent.position, goal)
        if path:
            agent.path = path
            agent.current_path_index = 0
            agent.reserved_cells = {}
            
            # Reserve cells along the path
            for timestep, cell in enumerate(path, start=self.current_time):
                self.grid_map.reserve_cell(cell[0], cell[1], timestep, agent.agent_id)
                agent.reserved_cells[timestep] = cell
            
            logger.info(f"Planned path for agent {agent.agent_id} to {goal}")
    
    def move_agent(self, agent: Agent):
        """Move an agent along its path if possible."""
        if not agent.path or agent.current_path_index >= len(agent.path):
            return False
        
        next_cell = agent.path[agent.current_path_index]
        next_timestep = self.current_time + 1
        
        # Check if next cell is available
        if self.grid_map.is_cell_reserved(next_cell[0], next_cell[1], next_timestep):
            reserved_agent_id = self.grid_map.reservation_table[next_timestep][next_cell]
            if reserved_agent_id != agent.agent_id:
                # Cell is reserved by another agent, wait
                agent.waiting_time += 1
                logger.debug(f"Agent {agent.agent_id} waiting at {agent.position}")
                return False
        
        # Move to next cell
        agent.position = next_cell
        agent.current_path_index += 1
        agent.total_distance += 1
        
        # Check if agent reached pickup or delivery point
        if agent.current_task:
            if agent.status == "MOVING_TO_PICKUP" and agent.position == agent.current_task.package_location:
                agent.status = "MOVING_TO_DELIVERY"
                agent.current_task.status = TaskStatus.IN_PROGRESS
                logger.info(f"Agent {agent.agent_id} picked up package at {agent.position}")
                # Replan path to delivery location
                self.plan_path_for_agent(agent)
            
            elif agent.status == "MOVING_TO_DELIVERY" and agent.position == agent.current_task.delivery_location:
                logger.info(f"Agent {agent.agent_id} delivered package at {agent.position}")
                self.completed_tasks.append(agent.current_task)
                agent.clear_task()
        
        return True
    
    def update(self):
        """Update the simulation by one timestep."""
        self.current_time += 1
        
        # Generate new tasks randomly
        if random.random() < 0.1:  # 10% chance each timestep
            self.generate_task()
        
        # Allocate tasks to idle agents
        self.allocate_tasks()
        
        # Plan paths for agents with tasks but no path
        for agent in self.agents:
            if agent.current_task and not agent.path:
                self.plan_path_for_agent(agent)
        
        # Replan paths periodically
        if self.current_time % self.replanning_frequency == 0:
            for agent in self.agents:
                if agent.current_task and agent.path:
                    # Clear old reservations
                    for timestep, cell in agent.reserved_cells.items():
                        self.grid_map.clear_reservation(cell[0], cell[1], timestep)
                    
                    # Replan path
                    self.plan_path_for_agent(agent)
        
        # Move agents
        for agent in self.agents:
            self.move_agent(agent)
        
        # Clean up completed tasks
        self.tasks = [t for t in self.tasks if t.status != TaskStatus.COMPLETED]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current simulation metrics."""
        completed = len(self.completed_tasks)
        pending = len([t for t in self.tasks if t.status == TaskStatus.PENDING])
        assigned = len([t for t in self.tasks if t.status == TaskStatus.ASSIGNED])
        in_progress = len([t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS])
        
        total_waiting = sum(agent.waiting_time for agent in self.agents)
        total_distance = sum(agent.total_distance for agent in self.agents)
        total_tasks_completed = sum(agent.tasks_completed for agent in self.agents)
        
        avg_waiting = total_waiting / len(self.agents) if self.agents else 0
        avg_distance = total_distance / len(self.agents) if self.agents else 0
        
        return {
            "completed_tasks": completed,
            "pending_tasks": pending,
            "assigned_tasks": assigned,
            "in_progress_tasks": in_progress,
            "total_waiting_time": total_waiting,
            "total_distance": total_distance,
            "total_tasks_completed": total_tasks_completed,
            "avg_waiting_time": avg_waiting,
            "avg_distance": avg_distance
        }


class MultiAgentVisualization:
    """Handles visualization of the multi-agent simulation."""
    
    def __init__(self, simulator: SwarmSimulator):
        self.simulator = simulator
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.3)
        
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup the matplotlib visualization."""
        self.ax.set_xlim(0, self.simulator.grid_map.width)
        self.ax.set_ylim(0, self.simulator.grid_map.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Multi-Agent Warehouse Simulation', fontsize=14)
        
        # Draw obstacles
        obstacle_x, obstacle_y = np.where(self.simulator.grid_map.obstacles)
        self.obstacles_plot = self.ax.scatter(obstacle_x, obstacle_y, color='black', 
                                             marker='s', s=50, label='Obstacles')
        
        # Initialize agent plots
        self.agent_plots = []
        self.agent_path_plots = []
        self.task_plots = []
        
        for agent in self.simulator.agents:
            agent_plot, = self.ax.plot([], [], 'o', color=agent.color, 
                                      markersize=10, label=f'Agent {agent.agent_id}')
            path_plot, = self.ax.plot([], [], '-', color=agent.color, alpha=0.5, linewidth=2)
            self.agent_plots.append(agent_plot)
            self.agent_path_plots.append(path_plot)
        
        # Add control buttons
        self.play_button_ax = plt.axes([0.7, 0.15, 0.1, 0.04])
        self.play_button = Button(self.play_button_ax, 'Start')
        self.play_button.on_clicked(self.on_play)
        
        self.reset_button_ax = plt.axes([0.81, 0.15, 0.1, 0.04])
        self.reset_button = Button(self.reset_button_ax, 'Reset')
        self.reset_button.on_clicked(self.on_reset)
        
        self.step_button_ax = plt.axes([0.7, 0.09, 0.1, 0.04])
        self.step_button = Button(self.step_button_ax, 'Step')
        self.step_button.on_clicked(self.on_step)
        
        # Add speed control slider
        self.speed_slider_ax = plt.axes([0.1, 0.15, 0.3, 0.03])
        self.speed_slider = Slider(self.speed_slider_ax, 'Speed', 100, 1000, valinit=500)
        
        # Add allocation strategy selector
        self.strategy_ax = plt.axes([0.1, 0.05, 0.2, 0.08])
        self.strategy_radio = RadioButtons(self.strategy_ax, ['Greedy', 'Auction'])
        self.strategy_radio.on_clicked(self.on_strategy_change)
        
        # Add metrics text
        self.metrics_text = self.ax.text(0.5, -0.1, "", transform=self.ax.transAxes, 
                                        ha='center', va='center', fontsize=10)
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Animation
        self.animation = None
        self.update_visualization()
    
    def on_strategy_change(self, label):
        """Handle strategy change event."""
        if label == 'Greedy':
            self.simulator.allocation_strategy = AllocationStrategy.GREEDY_NEAREST
        else:
            self.simulator.allocation_strategy = AllocationStrategy.AUCTION
        self.update_status_text(f"Strategy changed to: {label}")
    
    def on_play(self, event):
        """Handle play button click."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            self.play_button.label.set_text('Start')
        else:
            interval = self.speed_slider.val
            self.animation = FuncAnimation(
                self.fig, self.update_animation, interval=interval, blit=False
            )
            self.play_button.label.set_text('Pause')
        plt.draw()
    
    def on_step(self, event):
        """Handle step button click."""
        self.simulator.update()
        self.update_visualization()
        plt.draw()
    
    def on_reset(self, event):
        """Handle reset button click."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            self.play_button.label.set_text('Start')
        
        # Reset simulator
        grid_map = self.simulator.grid_map
        num_agents = self.simulator.num_agents
        strategy = self.simulator.allocation_strategy
        replanning = self.simulator.replanning_frequency
        
        self.simulator = SwarmSimulator(grid_map, num_agents, strategy, replanning)
        self.update_visualization()
        self.update_status_text("Simulation reset")
        plt.draw()
    
    def update_animation(self, frame):
        """Update function for animation."""
        self.simulator.update()
        self.update_visualization()
        return self.agent_plots + self.agent_path_plots + [self.metrics_text]
    
    def update_visualization(self):
        """Update the visualization with current simulator state."""
        # Update agent positions
        for i, agent in enumerate(self.simulator.agents):
            self.agent_plots[i].set_data([agent.position[0]], [agent.position[1]])
            
            # Update agent paths
            if agent.path:
                path_x, path_y = zip(*agent.path)
                self.agent_path_plots[i].set_data(path_x, path_y)
            else:
                self.agent_path_plots[i].set_data([], [])
        
        # Update tasks visualization
        # Remove existing task plots
        for plot in self.task_plots:
            plot.remove()
        self.task_plots = []
        
        # Draw package locations
        for task in self.simulator.tasks:
            if task.status == TaskStatus.PENDING:
                color = 'gray'
            elif task.status == TaskStatus.ASSIGNED:
                color = self.simulator.agents[task.assigned_agent.agent_id].color
            elif task.status == TaskStatus.IN_PROGRESS:
                color = 'orange'
            
            package_plot = self.ax.plot(task.package_location[0], task.package_location[1], 
                                       's', color=color, markersize=8, alpha=0.7)[0]
            delivery_plot = self.ax.plot(task.delivery_location[0], task.delivery_location[1], 
                                        'D', color=color, markersize=8, alpha=0.7)[0]
            self.task_plots.extend([package_plot, delivery_plot])
        
        # Update metrics
        metrics = self.simulator.get_metrics()
        metrics_text = (
            f"Time: {self.simulator.current_time} | "
            f"Completed: {metrics['completed_tasks']} | "
            f"Pending: {metrics['pending_tasks']} | "
            f"Assigned: {metrics['assigned_tasks']} | "
            f"In Progress: {metrics['in_progress_tasks']}"
        )
        self.metrics_text.set_text(metrics_text)
    
    def update_status_text(self, text: str):
        """Update the status text."""
        self.metrics_text.set_text(text)
    
    def show(self):
        """Show the visualization."""
        plt.show()


def main():
    """Main function to run the simulation."""
    print("Multi-Agent Warehouse Simulation")
    print("================================")
    print("Agents automatically navigate to complete delivery tasks")
    print("Controls:")
    print("- Start/Pause: Begin or pause simulation")
    print("- Step: Advance simulation by one timestep")
    print("- Reset: Reset simulation with new agents and tasks")
    print("- Speed: Adjust simulation speed")
    print("- Strategy: Select task allocation approach")
    print("  - Greedy: Assign tasks to nearest available agent")
    print("  - Auction: Agents bid for tasks based on distance")
    
    # Create map
    grid_map = GridMap(20, 20)
    grid_map.generate_random_obstacles(0.2)
    
    # Create simulator
    simulator = SwarmSimulator(grid_map, num_agents=5)
    
    # Create visualization
    visualization = MultiAgentVisualization(simulator)
    
    # Show the visualization
    visualization.show()


if __name__ == "__main__":
    main()
        
 
 