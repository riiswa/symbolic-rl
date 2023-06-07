# Symbolic Q-Learning

A reinforcement learning algorithm that implements Symbolic Q-Learning as presented in the paper:

Chloé Mercier, Frédéric Alexandre, Thierry Viéville. Reinforcement Symbolic Learning. ICANN 2021
- 30th International Conference on Artificial Neural Networks, Sep 2021, Bratislava / Virtual, Slovakia.
ffhal-03327706f
https://hal.inria.fr/hal-03327706/document.

## Algorithm

The SymbolicQLearning algorithm extends the QLearning algorithm by including a distance function in the update rule. The distance function is used to weight the contribution of the observed transition to the Q-value update, based on how close the new state is to the previous state. The closer the new state is, the higher the weight given to the transition, which allows the algorithm to adjust its state-action values more quickly.

## Installation

Requires Python 3.10 and installation of the dependencies with `pip install -r requirements.txt`.

```
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

python agents.py \\ Run agents
python SymbolicEnv.py \\ Run random agents

tensorboard --logdir=runs/ //localhost:6006
```

## Symbolic Environment

The symbolic environment contains a Gym environment that is connected to an ontology, and contains manually defined distances between each state. 

```python
env = SymbolicEnv()

observation = env.reset()

for i in range(1000):
    observation, reward, terminated, info = env.step(env.action_space.sample())

    if terminated:
        break
env.close()
```


## Symbolic Agent

The symbolic agent implements the paper formula in an efficient way (using vector operation and cache). To use it, you need to give it a function that takes two states of an environment and returns a distance (or similarity). During training you will need to define a radius. The agent is totally independent of the symbolic environment (defined just for testing). The agent uses Tensorboard to monitor the evolution of the reward at each iteration.

```python
env = SymbolicEnv()

model = SymbolicQLearning(env, env.dist)
model.learn(total_timesteps=10000, learning_rate=0.1, discount_factor=0.5, radius=0.001, log_name=f"runs/symbolic-ql-{r}")
```

