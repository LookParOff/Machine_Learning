import gym
import numpy as np


def train(Q_table, obs, n_episodes=50000, lr=0.7, gamma=0.9):
    epsilon = 1
    min_epsilon = 0.1
    max_epsilon = 1
    epsilon_discounter = 0.00002

    max_step = Q_table.shape[0] * 2
    step = 0

    render_ep = n_episodes - 3
    done = False

    for ep in range(n_episodes):
        while not done:
            action = np.argmax(Q_table[obs])
            if np.random.random() < epsilon and ep < render_ep:
                action = np.random.randint(0, 4)  # perform random action only if not render
            new_obs, reward, done, info = env.step(action)
            # if done and step < max_step:  # if fall in hole
            #     reward = -0.001
            Q_table[obs][action] += lr * (reward + gamma * max(Q_table[new_obs]) - Q_table[obs][action])
            obs = new_obs
            step += 1
            if step > max_step:
                done = True
                step = 0
            if ep >= render_ep:
                env.render()
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-epsilon_discounter * ep)
        obs = env.reset()
        done = False
    env.close()
    return Q_table


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    print(f"Observation Space {env.observation_space}")
    print(f"Action Space {env.action_space}")
    n_rows = env.observation_space.n
    n_cols = env.action_space.n
    Q_table_trained = np.zeros((n_rows, n_cols))
    observation = env.reset()

    Q_table_trained = train(Q_table_trained, observation)

