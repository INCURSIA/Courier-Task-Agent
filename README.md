Project: SARSA vs Q-learning
This project compares two fundamental Reinforcement Learning (RL) algorithms: SARSA (State-Action-Reward-State-Action) and Q-learning. Both algorithms are used to train an agent in a custom-built grid-world environment, where the agent interacts with different zones (like warehouse, building, and red light) to maximize its cumulative reward.

Key Features:
SARSA (On-policy): Updates Q-values based on the current action taken by the agent, which makes it slower to converge but provides more control over the exploration-exploitation tradeoff.

Q-learning (Off-policy): Updates Q-values based on the maximum possible reward, which allows faster convergence and better generalization but may be less stable in some environments.

The project compares these algorithms in terms of:

Convergence speed

Cumulative reward

Stability and generalization

The results highlight the strengths and weaknesses of each algorithm and provide insights into their performance in different types of RL environments.

```bash
#run
python main.py