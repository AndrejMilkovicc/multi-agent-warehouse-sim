Multi-Agent Warehouse Simulation
Description

This project implements a 2D warehouse simulation with multiple autonomous agents. The agents move across a grid, pick up and deliver packages, while using the A* path planning algorithm with reservation tables to avoid collisions.

Key Features

Multiple agents with different colors and tasks

Centralized task allocation (greedy nearest or auction-based)

Collision avoidance using reservation tables

Real-time visualization with agent paths and task status

Performance monitoring and simulation metrics

How to Run

Install the required libraries:

pip install matplotlib numpy


Run the script:

python fulglupo.py


Use the GUI controls to manage the simulation:

Start/Pause – start or pause the simulation

Step – advance the simulation by one step

Reset – reset the simulation with new agents and tasks

Speed – adjust the simulation speed

Strategy – select the task allocation method

Greedy – assigns the task to the nearest available agent

Auction – agents compete with bids based on distance

Code Structure

Task – represents a package pickup and delivery task

Agent – models an agent executing tasks

GridMap – defines the grid, obstacles, and cell reservations

AStarPlanner – implementation of the A* path planning algorithm

SwarmSimulator – central simulator managing agents and tasks

MultiAgentVisualization – simulation visualization and control panel

Simulation Metrics

The simulator continuously tracks:

number of completed, assigned, active, and pending tasks

total and average waiting time of agents

total and average distance traveled by agents

number of tasks completed per agent
