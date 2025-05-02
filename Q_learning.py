from environment import rows, columns, actions, movement, get_valid_start
import numpy as np
import random

Q = np.zeros((rows, columns, 2, 3, len(actions)))
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episode_rewards = []

def greedy_epsilon(state):
    if random.uniform(0,1) < epsilon:
        return random.choice(range(len(actions)))
    else:
        return np.argmax(Q[state[0],state[1],state[2],state[3],:])

def update_rule(episodes):
    for episode in range(episodes):
        man_row, man_col = (3, 0)
        package_status = 0
        check_point_status = 0
        state = (man_row, man_col, package_status, check_point_status)
        Done = False
        total_reward = 0
        while not Done:
            action = greedy_epsilon(state)
            next_state, reward, Done = movement(state, action)
            total_reward += reward
            Q[state[0],state[1],state[2],state[3],action] += alpha * (
                reward + gamma * np.max(Q[next_state[0],next_state[1],next_state[2],next_state[3],:])
                - Q[state[0],state[1],state[2],state[3],action]
            )
            state = next_state
        episode_rewards.append(total_reward)
    return Q,episode_rewards 
