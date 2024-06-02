#IMPORTING LIBRARIES
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import streamlit as st
import trimesh

#///////////////////
#///////////////////

#GETTING DOODLE
def get_doodle_coordinates():
    st.title("Doodle to Coordinates")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fill color with some opacity
        stroke_width=2,
        stroke_color="black",
        background_color="white",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

    coordinates = []
    
    # Process the canvas image
    if canvas_result.image_data is not None:
        # Convert the image to grayscale
        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
        image_np = np.array(image)

        # Threshold the image
        _, binary_image = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract coordinates
        for contour in contours:
            for point in contour:
                coordinates.append((point[0][0], point[0][1]))

    return coordinates

# To run Streamlit app:
if __name__ == '__main__':
    coordinates = get_doodle_coordinates()
    st.write("Coordinates:", coordinates)



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

# Convert grid to mesh
def grid_to_mesh(grid):
    vertices = []
    faces = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                vertices.append([x, y, 0])
    return trimesh.Trimesh(vertices=vertices, faces=faces)

# Streamlit app
st.title("3D Object Generator")

# Doodle input (file uploader)
uploaded_file = st.file_uploader("Upload a doodle", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    coordinates = extract_coordinates(image)
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
    mesh = grid_to_mesh(ca_grid)
    mesh.export('output.obj')
    st.write("3D model saved as output.obj")

    # Optionally display the mesh
    st.write("Generated 3D Mesh:")
    vertices = np.array(mesh.vertices)
    st.write(vertices)