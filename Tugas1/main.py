import numpy as np

def k_armed_testbed(k=10, num_problems=2000):
    """
    Generates a set of k-armed bandit problems.

    Args:
        k: The number of arms (actions).
        num_problems: The number of bandit problems to generate.

    Returns:
        A list of numpy arrays, where each array represents a problem
        and contains the true action values (q*(a)).
    """

    problems = []
    for _ in range(num_problems):
        # Generate true action values from a normal distribution with mean 0 and variance 1
        action_values = np.random.normal(0, 1, k)
        problems.append(action_values)
    return problems


def get_reward(action_values, chosen_action):
    """
    Returns a reward drawn from a normal distribution with mean q*(A) and variance 1.

    Args:
        action_values: A numpy array of true action values.
        chosen_action: The index of the chosen action (0-indexed).

    Returns:
        The reward.
    """

    mean_reward = action_values[chosen_action]
    reward = np.random.normal(mean_reward, 1)
    return reward


# Example usage: Generate 2000 10-armed bandit problems
k = 10
num_problems = 2000
problems = k_armed_testbed(k, num_problems)

# Example: Simulate one step in a single problem
problem_index = 0  # Choose a problem from the generated set
action_values = problems[problem_index]

chosen_action = 3 # Example action (remember actions are 0-indexed, so this is the 4th action)
reward = get_reward(action_values, chosen_action)