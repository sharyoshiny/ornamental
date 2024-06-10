#IMPORTING LIBRARIES
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid



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

# Multi-Agent System for Path Optimization
class PathAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
    
    def step(self):
        # Implement path optimization logic here
        pass

class PathOptimizationModel(Model):
    def __init__(self, width, height):
        self.num_agents = 10  # Define the number of agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Add agents to the model
        for i in range(self.num_agents):
            agent = PathAgent(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
    
    def step(self):
        self.schedule.step()

def extract_path_from_model(model):
    # Implement the logic to extract the optimized path from the model
    # This function is a placeholder and needs your implementation
    return [(0, 0), (1, 1), (2, 2)]  # Example path

def optimize_path(doodle_path):
    model = PathOptimizationModel(10, 10)
    for i in range(100):
        model.step()
    # Extract optimized path from model
    optimized_path = extract_path_from_model(model)
    return optimized_path

# Cellular Automata for Shape Generation
class CellularAutomata:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=int)
    
    def initialize_from_path(self, path):
        for (x, y) in path:
            self.grid[x, y] = 1
    
    def step(self):
        new_grid = self.grid.copy()
        # Implement CA rules here
        self.grid = new_grid
    
    def run(self, steps):
        for _ in range(steps):
            self.step()
    
    def get_shapes(self):
        return self.grid

def generate_shapes(path):
    ca = CellularAutomata(width=100, height=100)
    ca.initialize_from_path(path)
    ca.run(steps=50)
    shapes = ca.get_shapes()
    return shapes

# Main function to integrate everything
def main():
    coordinates = get_doodle_coordinates()
    st.write("Coordinates:", coordinates)

    if coordinates:
        optimized_path = optimize_path(coordinates)
        shapes = generate_shapes(optimized_path)
        
        # Display the shapes using Streamlit
        st.write("Optimized Path:", optimized_path)
        st.image(shapes, caption='Generated Shapes', use_column_width=True)

if __name__ == "__main__":
    main()
