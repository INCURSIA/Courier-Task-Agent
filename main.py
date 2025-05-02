import matplotlib.pyplot as plt
from Q_learning import update_rule as q_learning_update_rule
from Sarsa import sarsa_update_rule
from test import test_agent
import numpy as np

def moving_average(data, window_size=100):
    return [np.mean(data[max(0, i - window_size):i+1]) for i in range(len(data))]

def main():
    episodes = 50000

    print("Training Q-learning...")
    Q_q, rewards_q = q_learning_update_rule(episodes=episodes)

    print("Training SARSA...")
    Q_s, rewards_s = sarsa_update_rule(episodes=episodes)

    print("\nTesting Q-learning Agent:")
    test_agent(Q_q, max_steps=100)

    print("\nTesting SARSA Agent:")
    test_agent(Q_s, max_steps=100)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average(rewards_q), label='Q-learning', color='blue')
    plt.plot(moving_average(rewards_s), label='SARSA', color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("SARSA vs Q-learning: Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
