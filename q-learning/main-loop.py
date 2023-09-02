import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)
observation, info = env.reset()

win_pct_list = []
scores = []
agent = Agent(0.001, 0.9, 1.0, 0.01)
n_episodes = 500000


for i in range(n_episodes):
    terminated, truncated = False, False
    observation, info = env.reset()
    score = 0

    while not terminated or not truncated:
        action = None
        if np.random.random() < agent.current_epsilon:
            action = np.random.randint(0,4)
        else:
            #action = np.argmax(agent.Q[observation].values())
            action = max(agent.Q[observation], key=agent.Q[observation].get)

        if agent.current_epsilon > agent.epsilon_min:
            agent.current_epsilon = agent.current_epsilon * 0.99999995

        observation_, reward, terminated, truncated, info = env.step(action)

        #agent.Q[observation][action] += agent.step_size * (reward + (agent.discount_factor * np.argmax(agent.Q[observation_].values())) - agent.Q[observation][action])
        best_next_action = max(agent.Q[observation_], key=agent.Q[observation_].get)
        agent.Q[observation][action] += agent.step_size * (reward + (agent.discount_factor * agent.Q[observation_][best_next_action]) - agent.Q[observation][action])

        score += reward
        observation = observation_

    scores.append(score)

    if i % 100 == 0:
        win_pct = np.mean(scores[-100:])
        win_pct_list.append(win_pct)

        if i % 1000 == 0:
            print('Episode ', i, 'win pct %.2f' % win_pct, 'epsilon %.2f' % agent.current_epsilon)

print(agent.Q)
    
plt.plot(win_pct_list)
plt.show()