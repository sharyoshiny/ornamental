#IMPORTING LIBRARIES
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from stl import mesh
from streamlit_drawable_canvas import st_canvas

#///////////////////
#///////////////////

#GETTING DOODLE


# Function to extract coordinates from doodle
def extract_coordinates(image):
    image = np.array(image.convert('L'))  # Convert to grayscale
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    coordinates = np.column_stack(np.where(binary_image.transpose() > 0))
    return coordinates

# Agent class for Multi-Agent System
class Agent:
    def __init__(self, position):
        self.position = position

    def move(self, grid):
        x, y = self.position
        x += np.random.choice([-1, 0, 1])
        y += np.random.choice([-1, 0, 1])
        self.position = (x % grid.shape[0], y % grid.shape[1])

# Apply Multi-Agent System
def apply_multi_agent_system(grid, agents, steps):
    for step in range(steps):
        for agent in agents:
            agent.move(grid)
            x, y = agent.position
            grid[x, y] = 1  # Mark the agent's path on the grid
    return grid

# Initialize grid for Cellular Automata
def initialize_grid(size):
    return np.zeros(size)

# Apply Cellular Automata
def apply_cellular_automata(grid, steps):
    for step in range(steps):
        new_grid = grid.copy()
        for x in range(1, grid.shape[0] - 1):
            for y in range(1, grid.shape[1] - 1):
                live_neighbors = np.sum(grid[x-1:x+2, y-1:y+2]) - grid[x, y]
                if grid[x, y] == 1 and live_neighbors < 2 or live_neighbors > 3:
                    new_grid[x, y] = 0
                elif grid[x, y] == 0 and live_neighbors == 3:
                    new_grid[x, y] = 1
        grid = new_grid
    return grid

# Convert grid to mesh using numpy-stl
def grid_to_mesh(grid):
    vertices = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                vertices.append([x, y, 0])

    if len(vertices) < 3:
        return None

    vertices = np.array(vertices)
    faces = np.array([[i, (i + 1) % len(vertices), (i + 2) % len(vertices)] for i in range(len(vertices) - 2)])

    return mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

# Streamlit app
st.title("Doodle to 3D Object")

# Doodle input (canvas)
st.subheader("Doodle on the canvas below:")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=2,
    stroke_color="#000000",
    background_color="#ffffff",
    update_streamlit=True,
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    st.image(image, caption="Your Doodle")

    # Convert the canvas image to grayscale
    grayscale_image = image.convert("L")
    coordinates = extract_coordinates(grayscale_image)
    st.write("Coordinates extracted:", coordinates)

    # Save coordinates for further processing
    np.save('coordinates.npy', coordinates)

    # Load coordinates
    coordinates = np.load('coordinates.npy')

    # Generate 3D structure
    grid_size = (100, 100)
    grid = initialize_grid(grid_size)
    agents = [Agent((x, y)) for x, y in coordinates]
    final_grid = apply_multi_agent_system(grid, agents, 100)
    ca_grid = apply_cellular_automata(final_grid, 10)

    # Convert to mesh
    mesh_data = grid_to_mesh(ca_grid)
    if mesh_data:
        mesh_data.save('output.stl')
        st.write("3D model saved as output.stl")

        with open("output.stl", "rb") as file:
            st.download_button(
                label="Download 3D Model",
                data=file,
                file_name="output.stl",
                mime="application/octet-stream",
            )
    else:
        st.write("Failed to generate a valid 3D mesh.")