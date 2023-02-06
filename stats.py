import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class Stat:
    """
    Class for representing a single stat of the agent.

    Attributes:
    min (int): The minimum value that the stat can take.
    max (int): The maximum value that the stat can take.
    value (int): The current value of the stat.
    natural_increase (int): The amount by which the stat increases naturally after a certain number of steps.
    natural_increase_step (int): The number of steps after which the stat increases naturally.
    """
    min: int
    max: int
    value: int
    natural_increase: int
    natural_increase_step: int

    def update(self, i):
        """
        Update the value of the stat based on the given integer i.
        The new value is calculated by adding i to the current value.
        The resulting value is then clipped to the bounds of min and max.
        """
        self.value += i
        self.value = min(self.value, self.max)
        self.value = max(self.value, self.min)

    def discretize(self):
        """
        Discretize the value of the Stat instance into 4 bins.
        Returns an integer between 0 and 3, representing the bin that the value belongs to.
        """
        if self.value < self.max // 4:
            return 0
        elif self.value < self.max // 2:
            return 1
        elif self.value < 3 * self.max // 4:
            return 2
        else:
            return 3


class Stats:
    """
    This class creates a representation of the statistics of an agent, with default values for its six parameters:
    - energy,
    - health,
    - joy,
    - anger,
    - fear,
    - sadness.
    These are represented as instances of the Stat class and stored in a list self._stats.
    """
    def __init__(self):
        # TODO: This is hard-coded, we should change this in the future by loading the info from ontology
        self._energy = Stat(0, 20, 20, -1, 2)
        self._health = Stat(0, 20, 20, 1, 5)
        self._joy = Stat(0, 20, 20, 1, 10)
        self._anger = Stat(0, 10, 0, -1, 3)
        self._fear = Stat(0, 10, 0, -1, 3)
        self._sadness = Stat(0, 10, 0, -1, 3)
        self._stats = [self._energy, self._health, self._joy, self._anger, self._fear, self._sadness]

        self._distances = np.zeros((len(self), len(self)))

        for i in range(len(self)):
            for j in range(i+1, len(self)):
                d = np.linalg.norm(self.id_to_vector(i) - self.id_to_vector(j))
                self._distances[i, j] = d
                self._distances[j, i] = d
        self._distances = self._distances / self._distances.max()

    @lru_cache(maxsize=None)
    def vector_to_id(self, vector):
        """
        Convert a vector representation of a stat to a unique integer id.

        Parameters:
        vector (List[int]): A vector representation of a stat, with each element representing a level of the stat.

        Returns:
        int: A unique integer id representing the stat.
        """
        id = 0
        for i in range(4):
            id += vector[i] * 4 ** i
        return id

    @lru_cache(maxsize=None)
    def id_to_vector(self, id):
        """
        Convert a given ID to its corresponding vector representation.

        Args:
        - id (int): The ID to be converted.

        Returns:
        - np.array: The vector representation of the given ID.
        """
        vector = []
        for i in range(4):
            vector.append(id % 4)
            id = id // 4
        return np.array(vector[::-1])

    def dist(self, i, j):
        return self._distances[i, j]

    def __len__(self):
        return 256 # 4**4

    def energy(self):
        return self._energy.value * self._energy.discretize()

    def health(self):
        return self._health.value * self._health.discretize()

    def mood(self):
        return max(
            self._joy.value * self._joy.discretize() -
            self._anger.value * self._anger.discretize() -
            self._fear.value * self._fear.discretize() -
            self._sadness.value * self._sadness.discretize(),
            0
        )

    def update(self, effect):
        getattr(self, "_" + effect.gives.name).update(effect.hasEffectValue)

    def get_obs(self):
        obs = [s.discretize() for s in self._stats]
        return self.vector_to_id((obs[0], obs[1], obs[2], int(round((obs[3] + obs[4] + obs[5])/3))))

    def is_terminated(self):
        return self._health.value == 0 or self._energy.value == 0

    def reset(self, seed=None):
        random.seed(seed)
        self._energy.value = random.randint(5, 20)
        self._health.value = random.randint(5, 20)
        self._joy.value = random.randint(5, 20)
        self._fear.value = random.randint(0, 10)
        self._anger.value = random.randint(0, 10)
        self._sadness.value = random.randint(0, 10)

    def get_reward(self):
        return (self.energy() + self.health() + self.mood()) / 240

    def natural_increase_stats(self, epoch):
        for stat in self._stats:
            if epoch % stat.natural_increase_step == 0:
                stat.update(stat.natural_increase)