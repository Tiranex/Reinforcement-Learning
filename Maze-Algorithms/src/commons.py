#Commons.py
import numpy as np
import random

def render_maze(maze_array: np.array, cell_size: int=20):
    """
    Renders the maze based on the given maze array.

    Args:
        maze_array (np.array): The maze represented as a 2D numpy array.
        cell_size (int, optional): The size of each cell in pixels. Defaults to 20.

    Returns:
        np.array: The rendered maze as a 3D numpy array representing an image.
    """
    
    # Define colors
    wall_color = (0, 0, 0)  # Black for walls
    open_color = (255, 255, 255)  # White for open cells
    player_color = (0, 0, 255)  # Red for player position
    goal_color = (255, 0, 0)  # Blue for goal position
    path_color = (0,255,0) # Dark red for player path

    # Get maze dimensions
    width = maze_array.shape[1]
    height = maze_array.shape[0]

    # Create a bigger maze image based on the previous one
    maze_image = np.ones((cell_size * height, cell_size * width, 3))
    
    # Draw maze cells
    for j in range(height):
        for i in range(width):
            x, y = i * cell_size, j * cell_size
            if maze_array[j, i] == "0":
                # Open cell
                maze_image[y:y + cell_size, x:x + cell_size] = open_color
            elif maze_array[j, i] == "W":
                # Wall
                maze_image[y:y + cell_size, x:x + cell_size] = wall_color
            elif maze_array[j, i] == "*":
                # Player path position
                maze_image[y:y + cell_size, x:x + cell_size] = path_color
            elif maze_array[j, i] == "G":
                # Goal position
                maze_image[y:y + cell_size, x:x + cell_size] = goal_color
            elif maze_array[j, i] == "P":
                # Player position
                maze_image[y:y + cell_size, x:x + cell_size] = player_color

    return maze_image

def create_maze(size_x: int, size_y:int, start_point: tuple=(0,0)):
    """
    Creates a maze using the depth-first search algorithm.

    Args:
        size_x (int): The number of rows in the maze.
        size_y (int): The number of columns in the maze.
        start_point (tuple, optional): The starting point of the maze. Defaults to (0,0).

    Returns:
        numpy.ndarray: The generated maze represented as a 2D numpy array.
    """
    
    maze=np.full((size_x,size_y), "W")
    
    # up right down left
    directions=[(-1,0), (0,1), (1,0), (0,-1)]

    maze[start_point]="0"
    stack=[start_point]

    while len(stack) > 0:
        # Get current node
        x,y = stack[-1]

        # Shuffle neighbour to explore
        random.shuffle(directions)

        # Start exploring
        for dx,dy in directions:
            # Do the displacement
            nx, ny = x+2*dx, y+2*dy
    
            # Check for nx and ny inside bounds and if there is wall on next move
            if nx >= 0 and ny >= 0 and nx < size_x and ny < size_y and maze[nx,ny]=="W":
                maze[nx,ny]="0"
                maze[x+dx,y+dy]="0"
                stack.append((nx,ny))
                movement_done=True
                break
            movement_done=False
        
        if not movement_done:
            stack.pop()

    return maze
        
    