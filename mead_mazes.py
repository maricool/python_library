import numpy as np
import matplotlib.pyplot as plt

import mead_general as mead

# TODO: Should make a 'maze' class, with associated: grid, actions
# TODO: Switch from x, y to rows, cols (because order is opposite)

# Parameters
gamma_def = 0.99
max_steps_enact = 100
num_attempts_random_search = 100
max_steps_random_search = 100

# Maze mapping from features to integers
maze_def = [
   'path',
   'wall',
   'exit',
   'init',
   'hell',
]

# Maze colors for plotting
# TODO: How to ensure the keys of this dict are the same as the features?
maze_colors = {
    'path': 'white',
    'wall': 'dimgrey',
    'exit': 'lightgreen',
    'init': 'bisque',
    'hell': 'red',
}

def add_maze_boundary(maze, top=True, sides=True):
    '''
    Add a boundary of walls to a maze
    '''
    nr, nc = maze.shape
    nr_ = nr+2 if top else nr
    nc_ = nc+2 if sides else nc
    maze_ = np.ones((nr_, nc_), dtype=int)*maze_def.index('wall')
    if top and sides:
        maze_[1:-1, 1:-1] = maze
    elif top:
        maze_[1:-1, :] = maze
    elif sides:
        maze_[:, 1:-1] = maze
    else:
        maze_ = maze
    return maze_

def plot_maze(maze, **kwargs):
    '''
    Make a plot of the maze
    '''
    from matplotlib import colors
    color_list = []
    for feature in maze_def:
        color = maze_colors[feature]
        color_list.append(color)
    cmap = colors.ListedColormap(color_list)
    fig, ax = plt.subplots(**kwargs)
    plt.xticks([]); plt.yticks([])
    plt.imshow(maze, cmap=cmap, vmin=0, vmax=len(maze_def)-1)
    return fig, ax

def create_reward_function(maze, rewards_def):
    '''
    Create the reward function for the maze
    '''
    rewards = np.zeros_like(maze, dtype=float)
    for i, tile in np.ndenumerate(maze):
        rewards[i] = rewards_def[maze_def[tile]]
    return rewards

def enumerate_actions(maze):
    '''
    Enumerate all possible actions in each state (coordinate) in the maze
    Needs to account for walls, etc.
    '''
    ny, nx = maze.shape
    actions = np.empty_like(maze, dtype=object)
    for (iy, ix), tile in np.ndenumerate(maze):
        tile_actions = []
        if tile == maze_def.index('wall'):
            tile_actions.append('')
        elif tile == maze_def.index('exit'):
            tile_actions.append('E')
        else:
            iR = mead.periodic_integer(ix+1, nx) # Tile to the right
            iL = mead.periodic_integer(ix-1, nx) # Tile to the left
            iU = mead.periodic_integer(iy+1, ny) # Tile above
            iD = mead.periodic_integer(iy-1, ny) # Tile below
            if maze[iy, iR] != maze_def.index('wall'): tile_actions.append('R')
            if maze[iy, iL] != maze_def.index('wall'): tile_actions.append('L')
            if maze[iU, ix] != maze_def.index('wall'): tile_actions.append('U')
            if maze[iD, ix] != maze_def.index('wall'): tile_actions.append('D')
            #print('Coorindates:', ix, iy)
            #print('right, left, up down:', iR, iL, iU, iD)
        actions[iy, ix] = tile_actions
    return actions

def move(x, y, nx, ny, action):
    '''
    Result of moving according to an action
    TODO: Do I really need the x_, y_ underscore variables?
    '''
    if action == 'R': 
        x_, y_ = mead.periodic_integer(x+1, nx), y
    elif action == 'L': 
        x_, y_ = mead.periodic_integer(x-1, nx), y
    elif action == 'U': 
        x_, y_ = x, mead.periodic_integer(y+1, ny)
    elif action == 'D': 
        x_, y_ = x, mead.periodic_integer(y-1, ny)
    else:
        print(action, x, y, nx, ny)
        print('Action:', action)
        raise ValueError('Action is not understood')
    #print(action, x, y, x_, y_, nx, ny)
    return (x_, y_)

def enact(policy, start_x, start_y, max_steps=100):
    '''
    Calculate the set of states (coordinates) associated with enacting a policy
    '''
    x = start_x; y = start_y
    xs = []; ys = []
    ny, nx = policy.shape
    for _ in range(max_steps):
        xs.append(x), ys.append(y)
        action = policy[y, x]
        if action == 'E': break
        x, y = move(x, y, nx, ny, action)
    return (xs, ys)

def calculate_value(coords, rewards, gamma=gamma_def):
    '''
    Calculate the value of enacting a particular policy
    Note that the policy has a particular starting point
    '''
    value = 0.; step = -1
    for x, y in zip(*coords): # TODO: Slow loop
        step += 1
        value += rewards[y, x]*gamma**step
    return value

def random_search_escape(actions, rewards, nx, ny, start_x, start_y, 
    seed=None, num_attempts=num_attempts_random_search, max_steps=max_steps_random_search):
    '''
    Use crude random-search method to stumble upon the best route out of a maze
    '''
    import random
    random.seed(seed)
    attempts_x = []; attempts_y = []
    for _ in range(num_attempts):
        x = start_x; y = start_y
        xs = []; ys = []
        for _ in range(max_steps):
            xs.append(x); ys.append(y)
            action = random.choice(actions[y, x])
            if action == 'E': break
            x, y = move(x, y, nx, ny, action)
        attempts_x.append(xs); attempts_y.append(ys)
    values = []
    for xs, ys in zip(attempts_x, attempts_y):
        value = calculate_value((xs, ys), rewards)
        values.append(value)
    index = np.argmax(values)
    value = values[index]
    xs = attempts_x[index]; ys = attempts_y[index]
    return (xs, ys), value

def create_random_policy(actions):
    '''
    Create a policy entirely at random
    '''
    from random import choice as random_choice
    policy = np.zeros_like(actions, dtype=object)
    for i, action in np.ndenumerate(actions):
        policy[i] = random_choice(action)
    return policy