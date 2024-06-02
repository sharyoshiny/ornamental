#IMPORTING LIBRARIES
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
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

# Convert grid to image
def grid_to_image(grid):
    img_size = (grid.shape[0], grid.shape[1])
    img = Image.new('L', img_size, 0)
    draw = ImageDraw.Draw(img)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                draw.point((x, y), 255)
    return img

# Streamlit app
st.title("Doodle to 2D Image")

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

    # Convert to image
    output_image = grid_to_image(ca_grid)
    output_image_path = "output_image.png"
    output_image.save(output_image_path)
    st.image(output_image, caption="Generated Image")

    with open(output_image_path, "rb") as file:
        st.download_button(
            label="Download 2D Image",
            data=file,
            file_name="output_image.png",
            mime="image/png",
        )