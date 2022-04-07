import numpy as np
import matplotlib.pyplot as plt

import mead_general as mead

# TODO: Should make a 'maze' class, with associated: grid, actions

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
    nr, nc = maze.shape
    actions = np.empty_like(maze, dtype=object)
    for (ir, ic), tile in np.ndenumerate(maze):
        tile_actions = []
        if tile == maze_def.index('wall'):
            tile_actions.append('')
        elif tile == maze_def.index('exit'):
            tile_actions.append('E')
        else:
            iU = mead.periodic_integer(ir+1, nr) # Tile above
            iD = mead.periodic_integer(ir-1, nr) # Tile below
            iR = mead.periodic_integer(ic+1, nc) # Tile to the right
            iL = mead.periodic_integer(ic-1, nc) # Tile to the left
            if maze[ir, iR] != maze_def.index('wall'): tile_actions.append('R')
            if maze[ir, iL] != maze_def.index('wall'): tile_actions.append('L')
            if maze[iU, ic] != maze_def.index('wall'): tile_actions.append('U')
            if maze[iD, ic] != maze_def.index('wall'): tile_actions.append('D')
            #print('Coorindates:', ix, iy)
            #print('right, left, up down:', iR, iL, iU, iD)
        actions[ir, ic] = tile_actions
    return actions

def _move(r, c, nr, nc, action):
    '''
    Result of moving according to an action
    TODO: Do I really need the r_, c_ underscore variables?
    '''
    if action == 'R': 
        r_, c_ = r, mead.periodic_integer(c+1, nc)
    elif action == 'L': 
        r_, c_ = r, mead.periodic_integer(c-1, nc)
    elif action == 'U': 
        r_, c_ = mead.periodic_integer(r+1, nr), c
    elif action == 'D': 
        r_, c_ = mead.periodic_integer(r-1, nr), c
    else:
        #print(action, r, c, nr, nc)
        print('Action:', action)
        print('Row', r, 'of', nr)
        print('Col', c, 'of', nc)
        raise ValueError('Action is not understood')
    #print(action, x, y, x_, y_, nx, ny)
    return (r_, c_)

def enact(policy, start_c, start_r, max_steps=100): # TODO: Flip c <-> r
    '''
    Calculate the set of states (coordinates) associated with enacting a policy
    '''
    r = start_r; c = start_c
    rs = []; cs = []
    nr, nc = policy.shape
    for _ in range(max_steps):
        rs.append(r), cs.append(c)
        action = policy[r, c]
        if action == 'E': break
        r, c = _move(r, c, nr, nc, action)
    return (cs, rs) # TODO: Flip c <-> r

def calculate_value(coords, rewards, gamma=gamma_def):
    '''
    Calculate the value of enacting a particular policy
    Note that the policy has a particular starting point
    # TODO: Use r, c instead of x, y and flip
    '''
    value = 0.; step = -1
    for x, y in zip(*coords): # TODO: Slow loop
        step += 1
        value += rewards[y, x]*gamma**step
    return value

def random_search_escape(actions, rewards, nc, nr, start_c, start_r, # TODO: Flip c, r
    seed=None, num_attempts=num_attempts_random_search, max_steps=max_steps_random_search):
    '''
    Use crude random-search method to stumble upon the best route out of a maze
    '''
    import random
    random.seed(seed)
    attempts_r = []; attempts_c = []
    for _ in range(num_attempts):
        r = start_r; c = start_c
        rs = []; cs = []
        for _ in range(max_steps):
            rs.append(r); cs.append(c)
            action = random.choice(actions[r, c])
            if action == 'E': break
            r, c = _move(r, c, nr, nc, action)
        attempts_r.append(rs); attempts_c.append(cs)
    values = []
    for rs, cs in zip(attempts_r, attempts_c):
        value = calculate_value((cs, rs), rewards) # TODO: Flip
        values.append(value)
    index = np.argmax(values)
    value = values[index]
    rs = attempts_r[index]; cs = attempts_c[index]
    return (cs, rs), value

def create_random_policy(actions):
    '''
    Create a policy entirely at random
    '''
    from random import choice as random_choice
    policy = np.zeros_like(actions, dtype=object)
    for i, action in np.ndenumerate(actions):
        policy[i] = random_choice(action)
    return policy