#IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from doodling import get_doodle_coordinates  # Import the function

#///////////////////
#///////////////////


#GENERATING WITH THE COORDINATES
# Function to initialize the grid with coordinates
def initialize_grid(width, height, coordinates):
    grid = np.zeros((height, width))
    for x, y in coordinates:
        if 0 <= x < width and 0 <= y < height:
            grid[y, x] = 1
    return grid

# Function to update the grid using Cellular Automata rules
def update_grid(grid, rule="conway"):
    new_grid = grid.copy()
    for i in range(1, grid.shape[0] - 1):
        for j in range(1, grid.shape[1] - 1):
            total = int((grid[i, j] +
                         grid[i-1, j] + grid[i+1, j] +
                         grid[i, j-1] + grid[i, j+1] +
                         grid[i-1, j-1] + grid[i-1, j+1] +
                         grid[i+1, j-1] + grid[i+1, j+1]) / 255)
            if rule == "conway":
                # Conway's Game of Life rules
                if grid[i, j] == 255:
                    if total < 2 or total > 3:
                        new_grid[i, j] = 0
                else:
                    if total == 3:
                        new_grid[i, j] = 255
    return new_grid

# Function to simulate the Cellular Automata for a number of steps
def simulate_ca(grid, steps, rule="conway"):
    for _ in range(steps):
        grid = update_grid(grid, rule)
    return grid

# Agent class for Multi-Agent System
class DoodleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        new_position = (self.pos[0] + self.random.choice([-1, 0, 1]), self.pos[1] + self.random.choice([-1, 0, 1]))
        if 0 <= new_position[0] < self.model.grid.width and 0 <= new_position[1] < self.model.grid.height:
            self.model.grid.move_agent(self, new_position)

# Model class for Multi-Agent System
class DoodleModel(Model):
    def __init__(self, width, height, initial_coordinates, n_agents=50):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        for coord in initial_coordinates:
            agent = DoodleAgent(coord, self)
            self.grid.place_agent(agent, coord)
            self.schedule.add(agent)
        for i in range(n_agents):
            agent = DoodleAgent(i + len(initial_coordinates), self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()

# Example usage
coordinates = get_doodle_coordinates()  # Get coordinates from the doodle
width, height = 100, 100
grid = initialize_grid(width, height, coordinates)
ca_result_grid = simulate_ca(grid, 10)

# Use the result of CA as the initial state for MAS
initial_coordinates = [(i, j) for i in range(ca_result_grid.shape[0]) for j in range(ca_result_grid.shape[1]) if ca_result_grid[i, j] == 1]
model = DoodleModel(width, height, initial_coordinates, n_agents=100)
for i in range(50):
    model.step()

# Visualization
final_grid = np.zeros((width, height))
for agent in model.schedule.agents:
    final_grid[agent.pos] = 1

plt.imshow(final_grid, cmap='binary')
plt.savefig('final_output.png')
plt.show()