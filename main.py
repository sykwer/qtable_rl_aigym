import numpy as np
import gym

ENV_NAME = "FrozenLake-v0"
LEARNING_RATE = 0.8
GAMMA = 0.95
EPI_NUM = 10000


def main():
    env = gym.make(ENV_NAME)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    reward_list = []

    for i in range(EPI_NUM):
        state = env.reset()
        epi_reward = 0
        step = 0

        while step < 99:
            # env.render()

            step += 1
            action = np.argmax(q_table[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            next_state, reward, done, _ = env.step(action)
            q_table[state][action] = q_table[state, action] + LEARNING_RATE * (
                        reward + GAMMA * np.max(q_table[next_state, :]) - q_table[state, action])
            epi_reward += reward
            state = next_state

            if done:
                print(epi_reward)
                break
        reward_list.append(epi_reward)

    print(q_table)


if __name__ == "__main__":
    main()
