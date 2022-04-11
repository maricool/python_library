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

def _value_iteration(transitions, rewards, gamma=gamma_def, verbose=False):
    '''
    Use value iteration to find the best possible value
    Finally calculate the best policy, the one that gives the best possible value
    TODO: Is is always good to pick the first action in the max when many actions have the same reward?
    '''

    # Initial calculations
    values = np.zeros_like(rewards)
    actions = list(transitions.keys())
    number_of_states = len(values)

    # Iterate to find the best possible value
    # TODO: Is the .copy() necessary below? Values is not use again.
    new_values = values.copy(); old_values = np.ones_like(values); steps = 0
    #while old_values != new_values:
    while not np.array_equal(old_values, new_values):
        steps += 1 # Increment counter
        old_values = new_values.copy() # Set the old values to equal the previous new values
        for state in range(number_of_states):
            expected_future_rewards = {}
            for action in actions: # Find the expected future rewards associated with all possible actions
                expected_future_rewards[action] = np.dot(transitions[action][state, :], old_values)
            # Updated values are instant reward plus max of expectation of discounted future rewards
            # The maximum is taken over all possible actions
            # If many actions have the same reward then the first will be picked
            new_values[state] = rewards[state]+gamma*max(expected_future_rewards.values())
    if verbose: print('Best values found in stpes:', steps)

    # Now calculate the set of actions (policy) associated with the best reward
    best_values = new_values; best_policy = []
    for state in range(number_of_states):
        expected_future_rewards = {}
        for action in actions:
            expected_future_rewards[action] = np.dot(transitions[action][state, :], best_values)
        policy_item = max(expected_future_rewards, key=expected_future_rewards.get) # Action with best reward
        best_policy.append(policy_item)
    return best_policy

def _policy_iteration(transitions, rewards, gamma=gamma_def, rseed=None, verbose=False):
    '''
    Use policy iteration to find the best policy
    '''
    number_of_states = len(rewards)
    actions = transitions.keys()

    # Generate a random initial policy
    policy = create_random_policy(actions, number_of_states, rseed=rseed)
    if verbose: 
        print('Using policy iteration to find the best policy')
        print('Initial policy:', policy)

    # The loop
    new_policy = policy.copy(); old_policy = []; steps = 0
    while new_policy != old_policy:
        steps += 1 # Increment step counter
        old_policy = new_policy.copy()
        values = calculate_policy_values(old_policy, transitions, rewards, gamma) 
        expected_future_rewards = {}
        for action in actions:
            expected_future_rewards[action] = transitions[action].dot(values)
        #print('expected_future_rewards:', expected_future_rewards)

        new_policy = []
        for state in range(number_of_states):
            expected_future_rewards = {}
            for action in actions:
                expected_future_rewards[action] = expected_future_rewards[action][state]
            #print('expected_future_rewards_state:', i, expected_future_rewards_state)
            policy_item = max(expected_future_rewards, key=expected_future_rewards.get)
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
        best_policy = _value_iteration(transitions, rewards, gamma=gamma, verbose=verbose)
    else:
        print('Method:', method)
        raise ValueError('Unknown method for computing the best policy')
    return best_policy