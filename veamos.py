### IMPORT LIBRARIES

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import networkx as nx
import plotly.graph_objects as go


# Function to sample coordinates from the doodle
def sample_coordinates(doodle_coords, num_samples=100):
    if len(doodle_coords) <= num_samples:
        return doodle_coords
    return np.random.choice(doodle_coords, num_samples, replace=False).tolist()

# Function to generate paths using NetworkX
def generate_mas_path(sampled_coords):
    G = nx.grid_2d_graph(300, 150)  # Example grid size

    paths = []
    for start, goal in zip(sampled_coords, sampled_coords[1:] + [sampled_coords[0]]):
        try:
            path = nx.shortest_path(G, source=start, target=goal)
            paths.append(path)
        except nx.NetworkXNoPath:
            pass  # Handle the case where there is no path

    return paths

# Function for Cellular Automata
def cellular_automata(grid_size, mas_paths, num_iterations=10):
    grid = np.zeros(grid_size)

    # Initialize grid with MAS paths
    for path in mas_paths:
        for (x, y) in path:
            grid[x][y] = 1

    # Cellular Automata iterations
    for _ in range(num_iterations):
        new_grid = grid.copy()
        for i in range(1, grid_size[0] - 1):
            for j in range(1, grid_size[1] - 1):
                total = sum([grid[i - 1][j], grid[i + 1][j], grid[i][j - 1], grid[i][j + 1]])
                if grid[i][j] == 1:
                    if total < 2 or total > 3:
                        new_grid[i][j] = 0
                else:
                    if total == 3:
                        new_grid[i][j] = 1
        grid = new_grid

    return grid

# Function to visualize the generated structure using Plotly
def visualize_structure(grid):
    fig = go.Figure(data=[go.Surface(z=grid)])
    st.plotly_chart(fig)

# Main Streamlit app
st.title("Doodle to 3D Object")

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=2,
    stroke_color="#000",
    background_color="#fff",
    height=150,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Retrieve doodle coordinates
if canvas_result.json_data is not None:
    doodle_coords = canvas_result.json_data["objects"]
    coords = [(int(obj["left"]), int(obj["top"])) for obj in doodle_coords if "left" in obj and "top" in obj]

    sampled_coords = sample_coordinates(coords)
    st.write("Sampled Coordinates:", sampled_coords)

    mas_paths = generate_mas_path(sampled_coords)
    st.write("MAS Paths:", mas_paths)

    grid_size = (300, 150)
    ca_shape = cellular_automata(grid_size, mas_paths)
    st.write("Cellular Automata Shape:", ca_shape)

    visualize_structure(ca_shape)