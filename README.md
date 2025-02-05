# Taxi Navigation with Q-learning and SARSA

## Introduction

This project implements **Q-learning** and **SARSA** algorithms to solve a taxi navigation problem using the OpenAI `Gym` library. The goal is to train a model that minimizes the number of steps the taxi takes to pick up and drop off a passenger while avoiding illegal actions.

## Background

Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and aims to maximize the cumulative reward over time. Q-learning and SARSA are two popular RL algorithms used to learn the best actions to take in given states to maximize the cumulative reward.

## Challenges

- **Exploration and Exploitation**: The agent needs to explore new actions to find more rewarding strategies while exploiting known actions that yield high rewards.
- **Handling the exploration/exploitation trade-off**: Traditional search algorithms generally don’t handle this trade-off, whereas RL algorithms do.
- **Learning from the consequences of actions**: RL algorithms learn from the consequences of their actions, making them ideal for scenarios where it’s impractical to model the environment completely before making decisions.

### Solutions

- **Exploration and Exploitation**: I used ε-Greedy and Softmax policies to balance exploration and exploitation. ε-Greedy allows the agent to explore random actions with a probability of ε, while Softmax selects actions based on a probability distribution.
- **Exploration/Exploitation Trade-off**: By tuning the parameters ε (for ε-Greedy) and τ (for Softmax), I was able to find a balance that provided stable learning and improved performance.
- **Learning from Actions**: Both Q-learning and SARSA algorithms were implemented to update the Q-values based on the agent's actions and the resulting rewards. Q-learning, being an off-policy algorithm, updates Q-values using the maximum possible future reward, while SARSA, being an on-policy algorithm, updates Q-values based on the actions actually taken by the policy.


## Dataset

The dataset used in this project is the `Taxi-v3` environment from the OpenAI `Gym` library. The environment consists of a taxi that needs to pick up and drop off a passenger at designated locations. The state of the environment changes based on the taxi’s actions.

## Method

### Q-learning

Q-learning is an off-policy algorithm that updates its Q-values using the maximum possible future reward, independently of the agent’s actions.

Update rule:
$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime}) - Q(s, a)]$

### SARSA

SARSA (State-Action-Reward-State-Action) is an on-policy algorithm that updates its Q-values based on the actions actually taken by the policy, including the exploration steps.

Update rule:
$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s^{\prime}, a^{\prime}) - Q(s, a)]$

### Action Selection Policies

1. **Greedy**: The agent always chooses the action with the highest estimated reward.
2. **ε-Greedy**: The agent chooses the action with the highest estimated reward with probability $\(1 - \epsilon\)$ and a random action with probability $\(\epsilon\)$.
3. **Softmax**: Actions are selected according to probabilities proportional to the exponential of their estimated values.

## Results

The results show that the Softmax policy tends to offer better performance than ε-Greedy. Higher ε or τ values lead to higher instability. The performance of Q-learning and SARSA is similar with Softmax as the action selection policy. Q-learning performs better than SARSA with ε-Greedy as SARSA is more sensitive to the ε value.

## Requirements

- Python 3.11.9
- Gym library
- NumPy
- Matplotlib

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/ivanluk914/Taxi-Navigation-with-Q-learning-and-SARSA.git
   ```
   
2. Install the required packages:
   ```sh
   pip install numpy matplotib gym
   ```

3. Run the Jupyter notebook:
    ```sh
    jupyter notebook taxi_navigation_rl.ipynb
    ```

4. Train the models and visualize the results by executing the cells in the notebook.

## Discussion Note
For further discussion and analysis, refer to the `discussion note.pdf`.

## Acknowledgements
- The Gym library for providing the Taxi-v3 environment.
- [Geeks For Geeks](https://www.geeksforgeeks.org/differences-between-q-learning-and-sarsa/) for the detailed explanation of Q-learning and SARSA algorithms.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
