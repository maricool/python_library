import numpy as np

# Parameters
gamma_def = 0.99 # Default value of the discount factor

def calculate_policy_values(policy, transitions, rewards, gamma=gamma_def):
    '''
    Computes the value associated with each state when enacting the policy
    A set of linear equations must be solved here
    '''
    n = len(rewards)
    prob = np.empty((n, n))
    for (i, j), _ in np.ndenumerate(prob):
        prob[i, j] = transitions[policy[i]][i, j]
    V = np.linalg.solve(np.identity(n)-gamma*prob, rewards) # Linear equation solving
    return V

def create_random_policy(actions, n, rseed=None):
    '''
    Compute a random policy
    '''
    from random import seed, choice
    seed(rseed)
    return [choice(list(actions)) for _ in range(n)]

def _policy_iteration(transitions, rewards, gamma=gamma_def, rseed=None, verbose=False):
    '''
    Use policy iteration to find the best policy
    '''
    n = len(rewards)
    actions = transitions.keys()

    # Generate a random initial policy
    policy = create_random_policy(actions, n, rseed=rseed)
    if verbose: 
        print('Using policy iteration to find the best policy')
        print('Initial policy:', policy)

    # The loop
    new_policy = policy; old_policy = []; steps = 0
    while new_policy != old_policy:
        steps += 1 # Increment step counter
        old_policy = new_policy.copy()
        values = calculate_policy_values(old_policy, transitions, rewards, gamma) 
        expected_future_rewards = {}
        for action in actions:
            expected_future_rewards[action] = transitions[action].dot(values)
        #print('expected_future_rewards:', expected_future_rewards)
        new_policy = []
        for i in range(n):
            expected_future_rewards_state = {}
            for action in actions:
                expected_future_rewards_state[action] = expected_future_rewards[action][i]
            #print('expected_future_rewards_state:', i, expected_future_rewards_state)
            policy_item = max(expected_future_rewards_state, key=expected_future_rewards_state.get)
            new_policy.append(policy_item)
        if verbose: print('Policy:', steps, new_policy)
    
    best_policy = new_policy
    if verbose: 
        print('Best policy:', best_policy)
        print('Found in steps:', steps)
    return best_policy

def compute_best_policy(transitions, rewards, gamma=gamma_def, method='policy_iteration', rseed=None, verbose=False):
    '''
    Wraps both policy iteration and value iteration algorithms
    '''
    if method == 'policy_iteration':
        best_policy = _policy_iteration(transitions, rewards, gamma=gamma, rseed=rseed, verbose=verbose)
    elif method == 'value_iteration':
        # best_policy = _value_iteration()
        raise ValueError('Value iteration is not implemented yet')
    else:
        raise ValueError('Unknown method')
    return best_policy