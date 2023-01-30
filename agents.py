import random
from functools import lru_cache
from tqdm import tqdm

import gym
import torch
import numpy as np
from gym import Space
from gym.spaces import MultiDiscrete, Discrete
from stable_baselines3.common.base_class import BaseAlgorithm, SelfBaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback

from SymbolicEnv import SymbolicEnv


def get_dims(space: Space):
    if isinstance(space, MultiDiscrete):
        return [s.n for s in space]
    elif isinstance(space, Discrete):
        return [space.n]
    else:
        raise RuntimeError("Not supported space.")


class QLearningPolicy(BasePolicy):
    def __init__(self, observation_space: Space, action_space: Space):
        BasePolicy.__init__(self, observation_space, action_space)
        self.observation_space = observation_space
        self.action_space = action_space

        dims = []
        dims.extend(get_dims(observation_space))
        dims.extend(get_dims(action_space))

        self.qtable = torch.zeros(tuple(dims), requires_grad=False)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        observation = observation.squeeze()
        actions = self.qtable[tuple(observation)]
        actions_idx = torch.where(actions == actions.max())[0]
        return actions_idx[torch.randint(0, len(actions_idx), size=(1,))]


class QLearning(BaseAlgorithm):
    policy: QLearningPolicy

    def __init__(self, env):
        BaseAlgorithm.__init__(self, QLearningPolicy, env, 0, )
        self._setup_model()

    def _update(self, observation, action, next_observation, reward, terminated, **kwargs):
        observation = observation.squeeze()
        next_observation = next_observation.squeeze()
        if "learning_rate" in kwargs and "discount_factor" in kwargs:
            learning_rate = kwargs["learning_rate"]
            discount_factor = kwargs["discount_factor"]

            idx = tuple([*observation, action.item()])

            update_term = reward if terminated else \
                reward + discount_factor * self.policy.qtable[tuple(next_observation)].max() - \
                self.policy.qtable[idx]

            self.policy.qtable[idx] = self.policy.qtable[idx] + learning_rate * update_term
        else:
            raise RuntimeError("kwargs should contain learning_rate and discount_factor")

    def learn(self: SelfBaseAlgorithm, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", reset_num_timesteps: bool = True,
              progress_bar: bool = False, **kwargs) -> SelfBaseAlgorithm:

        epsilon_start = 1.0
        epsilon_end = 0.1
        observation = self.env.reset()
        for step in tqdm(range(total_timesteps)):
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (step / total_timesteps)
            if random.random() < epsilon:
                action = torch.tensor(self.env.action_space.sample(), requires_grad=False)
            else:
                action, _ = self.policy.predict(observation)
            next_observation, reward, terminated, info = env.step(action.squeeze())
            self._update(observation, action, next_observation, reward, terminated, **kwargs)
            observation = next_observation
            if terminated:
                observation = self.env.reset()

        return self

    def _setup_model(self) -> None:
        self.policy = QLearningPolicy(env.observation_space, env.action_space)


class SymbolicQLearning(QLearning):
    def __init__(self, env: gym.Env, distance_function):
        QLearning.__init__(self, env)
        self.distance_function = distance_function

    @lru_cache(maxsize=None)
    def _get_weights(self, observation):
        def dist(x, y):
            return self.distance_function((x, y), observation)

        return torch.exp(-torch.from_numpy(
            np.fromfunction(np.vectorize(dist), get_dims(self.env.observation_space), dtype=int)
        ))

    def _update(self, observation, action, next_observation, reward, terminated, **kwargs):
        observation = observation.squeeze()
        next_observation = next_observation.squeeze()
        action = action.squeeze()
        if "learning_rate" in kwargs and "discount_factor" in kwargs and "radius" in kwargs:
            learning_rate = kwargs["learning_rate"]
            discount_factor = kwargs["discount_factor"]
            radius = kwargs["radius"]

            weights = learning_rate * self._get_weights(tuple(observation)) / radius

            update_term = torch.tensor(reward) if terminated else \
                reward + discount_factor * self.policy.qtable[tuple(next_observation)].max() - \
                self.policy.qtable[..., action]

            self.policy.qtable[..., action] = \
                self.policy.qtable[..., action] + weights * update_term
        else:
            raise RuntimeError("kwargs should contain learning_rate, discount_factor and radius")


if __name__ == "__main__":

    env = SymbolicEnv()
    check_env(env)

    model = SymbolicQLearning(env, env.dist)
    model.learn(total_timesteps=1000, learning_rate=0.5, discount_factor=0.5, radius=1)

    print(model.policy.qtable)

    observation = env.reset()

    for i in range(1000):
        action, _states = model.predict(observation)
        observation, reward, terminated, info = env.step(action)
        print(i, [s.value for s in env.stats._stats], reward)

        if terminated:
            break
    env.close()
