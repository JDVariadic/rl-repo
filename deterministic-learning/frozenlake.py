import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True)
observation, info = env.reset()
'''
win_pct = []
scores = []

for i in range(1000):
    terminated, truncated = False, False
    observation, info = env.reset()
    score = 0
    while not terminated or not truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward

    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
    
#plt.plot(win_pct)
#plt.show()
'''


#SAMPLE RUN


win_pct = []
scores = []
'''

0: LEFT
1: DOWN
2: RIGHT
3: UP

SFFH
FHFF
FHFH
FFFG
'''
policy = {0:2, 1:2, 2:1, 3:1, 4:1, 5:2, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:2, 14:2, 15:2}
for i in range(1000):
    terminated, truncated = False, False
    observation, info = env.reset()
    score = 0
    while not terminated or not truncated:
        #action = env.action_space.sample()
        modified_action = policy[observation]
        observation, reward, terminated, truncated, info = env.step(modified_action)
        score += reward

    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)
    
plt.plot(win_pct)
plt.show()