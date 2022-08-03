from time import time
import gym
import numpy as np


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed=None):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state][:])
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def get_states_value(Q_table):
    """
    Return matrix with estimation value of each state
    :param Q_table:
    :return:
    """
    n_rows = int(np.sqrt(Q_table.shape[0]))
    state_values = np.zeros((n_rows, n_rows))
    for i_row in range(n_rows-1, -1, -1):
        for i_col in range(n_rows-1, -1, -1):
            curr_obs = i_row * n_rows + i_col
            if np.max(Q_table[curr_obs]) == np.min(Q_table[curr_obs]):
                continue
            for action in range(4):
                new_obs = get_expected_obs(curr_obs, action, n_rows)
                r, c = get_col_row(new_obs, n_rows)
                state_values[r][c] = Q_table[curr_obs][action]
    state_values[n_rows-1][n_rows-1] = 1
    return state_values


def get_col_row(obs, n_rows):
    """
    >>> print(1)
    1
    >>> get_col_row(0, 4)
    (0, 0)
    >>> get_col_row(1, 4)
    (0, 1)
    >>> get_col_row(3, 4)
    (0, 3)
    >>> get_col_row(4, 4)
    (1, 0)
    >>> get_col_row(8, 4)
    (2, 0)
    >>> get_col_row(12, 4)
    (3, 0)
    >>> get_col_row(15, 4)
    (3, 3)
    >>> get_col_row(63, 8)
    (7, 7)
    >>> get_col_row(59, 8)
    (7, 3)

    :param obs:
    :param n_rows:
    :return:
    """
    # obs = row * n_rows + col
    row = obs // n_rows
    col = obs - row * n_rows
    return row, col


def get_expected_obs(obs, action, n_rows):
    """
    >>> get_expected_obs(15, 0, 4)
    14
    >>> get_expected_obs(14, 2, 4)
    15
    >>> get_expected_obs(4, 1, 4)
    8
    >>> get_expected_obs(4, 3, 4)
    0

    >>> get_expected_obs(0, 0, 4)
    0
    >>> get_expected_obs(0, 3, 4)
    0
    >>> get_expected_obs(15, 1, 4)
    15
    >>> get_expected_obs(15, 2, 4)
    15
    """
    row, col = get_col_row(obs, n_rows)
    expect_row, expect_col = row, col
    if action == 0:  # left
        expect_col = max(0, col - 1)
    if action == 1:  # down
        expect_row = min(n_rows - 1, row + 1)
    if action == 2:  # right
        expect_col = min(n_rows - 1, col + 1)
    if action == 3:  # up
        expect_row = max(0, row - 1)
    expect_obs = expect_row * n_rows + expect_col
    return expect_obs


def get_made_action(curr_obs, next_obs, n_rows):
    """
    >>> get_made_action(15, 14, 4)
    0
    >>> get_made_action(14, 15, 4)
    2
    >>> get_made_action(4, 0, 4)
    3
    >>> get_made_action(0, 4, 4)
    1
    >>> get_made_action(7, 11, 4)
    1
    >>> get_made_action(10, 11, 4)
    2
    >>> get_made_action(15, 11, 4)
    3
    >>> get_made_action(11, 11, 4)
    2
    >>> get_made_action(0, 0, 4)
    3

    :param curr_obs:
    :param next_obs:
    :param n_rows:
    :return:
    """
    curr_row, curr_col = get_col_row(curr_obs, n_rows)
    next_row, next_col = get_col_row(next_obs, n_rows)
    if curr_obs == next_obs:  # action is go to the wall
        if curr_row == n_rows - 1:
            return 1
        if curr_row == 0:
            return 3
        if curr_col == n_rows - 1:
            return 2
        if curr_col == 0:
            return 0
    if next_row > curr_row:
        return 1  # down
    if next_row < curr_row:
        return 3  # up
    if next_col > curr_col:
        return 2  # right
    if next_col < curr_col:
        return 0  # left


def train(Q_table, obs, n_episodes, max_step, lr=0.7, gamma=0.9, n_render_episodes=0):
    global history_of_fittness
    n_rows = int(np.sqrt(Q_table.shape[0]))
    epsilon = 1
    min_epsilon = 0
    max_epsilon = 1
    epsilon_discounter = 0.00006

    render_ep = n_episodes - n_render_episodes
    for ep in range(n_episodes):
        done = False
        step = 0
        reward = 0
        while not done:
            if np.random.random() < epsilon and ep < render_ep:
                action = np.random.randint(0, 4)  # perform random action only if not render
            else:
                action = np.argmax(Q_table[obs])
            new_obs, reward, done, info = env.step(action)
            # expected_obs = get_expected_obs(obs, action, n_rows)
            # if expected_obs != new_obs:  # we slipped
            #     action = get_made_action(obs, new_obs, n_rows)
            Q_table[obs][action] += \
                lr * (reward + gamma * np.max(Q_table[new_obs]) - Q_table[obs][action])
            obs = new_obs
            step += 1
            if step > max_step:
                done = True
            if ep >= render_ep:
                env.render()
        if ep >= render_ep:
            print(f"finish with reward={reward}, in step={step}")
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-epsilon_discounter * ep)
        obs = env.reset()

        # if ep % 1000 == 0:
        #     mean_r, std_r = evaluate_agent(env, max_agent_steps, 1000, Q_table_trained)
        #     history_of_fittness.append(mean_r - std_r)
    env.close()
    return Q_table


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    print(f"Observation Space {env.observation_space}")
    print(f"Action Space {env.action_space}")
    n_rows_of_table = env.observation_space.n
    n_cols_of_table = env.action_space.n
    Q_table_trained = np.zeros((n_rows_of_table, n_cols_of_table))
    observation = env.reset()

    # max_agent_steps = n_rows * 10
    max_agent_steps = 90

    history_of_fittness = []
    time_start = time()
    Q_table_trained = train(Q_table_trained, observation, 100_000, max_agent_steps, lr=0.01,
                            n_render_episodes=0)
    print(f"end in {round(time() - time_start, 2)} secs")
    mean_reward, std_reward = evaluate_agent(env, max_agent_steps, 1000, Q_table_trained)
    print(mean_reward, np.round(std_reward, 2), np.round(mean_reward - std_reward, 2))
    print(get_states_value(Q_table_trained))
