import numpy as np
import matplotlib.pyplot as plt

import mead_general as mead

# TODO: Should make a 'maze' class, with associated: grid, actions
# TODO: Make periodic maze an option

# Parameters
gamma_def = 0.99                 # Discount factor
max_steps_enact = 100            #
num_attempts_random_search = 100 #
max_steps_random_search = 100    #

action_left = 'L'
action_right = 'R'
action_up = 'U'
action_down = 'D'
action_exit = 'E'

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
    'wall': 'green',
    'exit': 'lightgreen',
    'init': 'bisque',
    'hell': 'red',
}

def make_maze(w=16, h=8, double_gap=True, rseed=None):
    '''
    From: https://rosettacode.org/wiki/Maze_generation#Python
    '''
    from random import seed, shuffle, randrange

    # Text features
    wg = '|  ' if double_gap else '| '
    ww = '+--' if double_gap else '+-'
    cg = '+  ' if double_gap else '+ '
    gg = '   ' if double_gap else '  '
    wa = '|'; co = '+'

    vis = [[0]*w+[1] for _ in range(h)]+[[1]*(w+1)]
    ver = [[wg]*w+[wa] for _ in range(h)]+[[]]
    hor = [[ww]*w+[co] for _ in range(h+1)]

    seed(rseed)
    def walk(x, y):
        vis[y][x] = 1
        d = [(x-1, y), (x, y+1), (x+1, y), (x, y-1)]
        shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]: continue
            if xx == x: hor[max(y, yy)][x] = cg
            if yy == y: ver[y][max(x, xx)] = gg
            walk(xx, yy)

    walk(randrange(w), randrange(h))

    s = ''
    for (a, b) in zip(hor, ver):
        s += ''.join(a+['\n']+b+['\n'])
    return s

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
            tile_actions.append(action_exit)
        else:
            iU = mead.periodic_integer(ir+1, nr) # Tile above
            iD = mead.periodic_integer(ir-1, nr) # Tile below
            iR = mead.periodic_integer(ic+1, nc) # Tile to the right
            iL = mead.periodic_integer(ic-1, nc) # Tile to the left
            if maze[ir, iR] != maze_def.index('wall'): tile_actions.append(action_right)
            if maze[ir, iL] != maze_def.index('wall'): tile_actions.append(action_left)
            if maze[iU, ic] != maze_def.index('wall'): tile_actions.append(action_up)
            if maze[iD, ic] != maze_def.index('wall'): tile_actions.append(action_down)
            #print('Coorindates:', ix, iy)
            #print('right, left, up down:', iR, iL, iU, iD)
        actions[ir, ic] = tile_actions
    return actions

def _move(r, c, nr, nc, action):
    '''
    Result of moving according to an action
    TODO: Do I really need the r_, c_ underscore variables?
    '''
    if action == action_right: 
        r_, c_ = r, mead.periodic_integer(c+1, nc)
    elif action == action_left: 
        r_, c_ = r, mead.periodic_integer(c-1, nc)
    elif action == action_up: 
        r_, c_ = mead.periodic_integer(r+1, nr), c
    elif action == action_down: 
        r_, c_ = mead.periodic_integer(r-1, nr), c
    else:
        print('Action:', action)
        print('Row', r, 'of', nr)
        print('Col', c, 'of', nc)
        raise ValueError('Action is not understood')
    #print(action, x, y, x_, y_, nx, ny)
    return (r_, c_)

def enact(policy, start_r, start_c, max_steps=100):
    '''
    Calculate the set of states (coordinates) associated with enacting a policy
    '''
    r = start_r; c = start_c
    rs = []; cs = []
    nr, nc = policy.shape
    for _ in range(max_steps):
        rs.append(r), cs.append(c)
        action = policy[r, c]
        if action == action_exit: break
        r, c = _move(r, c, nr, nc, action)
    return (rs, cs)

def calculate_payoff(coords, rewards, gamma=gamma_def):
    '''
    Calculate the payoff of visiting a specific set of states in a certain order
    '''
    payoff = 0.; step = -1
    for r, c in zip(*coords): # TODO: Slow loop
        step += 1
        payoff += rewards[r, c]*gamma**step
    return payoff

def random_search_escape(actions, rewards, nr, nc, start_r, start_c,
    rseed=None, num_attempts=num_attempts_random_search, max_steps=max_steps_random_search):
    '''
    Use crude random-search method to stumble upon the best route out of a maze
    '''
    import random
    random.seed(rseed)
    attempts_r = []; attempts_c = []
    for _ in range(num_attempts):
        r = start_r; c = start_c
        rs = []; cs = []
        for _ in range(max_steps):
            rs.append(r); cs.append(c)
            action = random.choice(actions[r, c])
            if action == action_exit: break
            r, c = _move(r, c, nr, nc, action)
        attempts_r.append(rs); attempts_c.append(cs)
    payoffs = []
    for rs, cs in zip(attempts_r, attempts_c):
        payoff = calculate_payoff((rs, cs), rewards)
        payoffs.append(payoff)
    index = np.argmax(payoffs)
    payoff = payoffs[index]
    rs = attempts_r[index]; cs = attempts_c[index]
    return (rs, cs), payoff

def create_random_policy(actions):
    '''
    Create a policy entirely at random
    '''
    from random import choice as random_choice
    policy = np.zeros_like(actions, dtype=object)
    for i, action in np.ndenumerate(actions):
        policy[i] = random_choice(action)
    return policy

def learn_best_policy(maze, actions, rewards, verbose=False, gamma=gamma_def, rseed=None, method='policy_iteration'):

    # Use reinforcement learning to find the best policy
    import mead_reinforcement as reinforcement
    nr, nc = maze.shape

    # Count states, discarding wall tiles etc.
    # Create a mapping between the coordinates of tiles in the maze and the state number
    # Also calculate the rewards as a function of state in the same way
    state_map = []; rewards_map = []
    for (i, j), tile in np.ndenumerate(maze):
        if tile != maze_def.index('wall'):
            state_map.append((i, j))
            rewards_map.append(rewards[i, j])
    if verbose: mead.print_full_array(state_map, 'State map:')
    print()
    if verbose: mead.print_full_array(rewards_map, 'Rewards:')
    print()
    n = len(state_map)
    if verbose: print('Number of states:', n)

    # Calculate the effect of moving left, right, up or down in terms of state motions
    # TODO: These should probably be a dictionary with states['left'] etc.
    states_left = []; states_right = []; states_up = []; states_down = []
    for i in range(n): # Loop over states
        l = state_map.index((state_map[i][0], mead.periodic_integer(state_map[i][1]-1, nc))) if action_left in actions[state_map[i]] else i
        r = state_map.index((state_map[i][0], mead.periodic_integer(state_map[i][1]+1, nc))) if action_right in actions[state_map[i]] else i
        u = state_map.index((mead.periodic_integer(state_map[i][0]+1, nr), state_map[i][1])) if action_up in actions[state_map[i]] else i
        d = state_map.index((mead.periodic_integer(state_map[i][0]-1, nr), state_map[i][1])) if action_down in actions[state_map[i]] else i
        states_left.append(l); states_right.append(r); states_up.append(u); states_down.append(d)

    if verbose:
        print('States left:', states_left)
        print('States right:', states_right)
        print('States up:', states_up)
        print('States down:', states_down)
        print()

    # Compute transition matrices between the states
    # Because the motion of the robot is deterministic the rows all contain a single '1' and rest '0'
    transitions = {}
    for direction in [action_left, action_right, action_up, action_down]:
        transitions[direction] = np.zeros((n,n))
    for i in range(n):
        transitions[action_left][i, states_left[i]] = 1
        transitions[action_right][i, states_right[i]] = 1
        transitions[action_up][i, states_up[i]] = 1
        transitions[action_down][i, states_down[i]] = 1
    if verbose: print('Example transition matrix for left:\n', transitions[action_left], '\n')

    best_policy = reinforcement.compute_best_policy(transitions, rewards_map, 
        gamma=gamma, rseed=rseed, method=method, verbose=verbose)
    if verbose: print('Best policy:\n', best_policy, '\n')

    learned_policy = np.array(['']*nc*nr).reshape(nc, nr)
    for i, coords in enumerate(state_map):
        learned_policy[coords] = best_policy[i]
    er, ec = np.where(maze==maze_def.index('exit'))
    learned_policy[er, ec] = action_exit
    if verbose: print('Reformatted best policy:\n', learned_policy, '\n')
    return learned_policy