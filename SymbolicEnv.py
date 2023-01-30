import random
from dataclasses import dataclass
from typing import Any, Optional
import yaml

import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

import yamlpyowl as ypo
from owlready2 import *
from stable_baselines3.common.env_checker import check_env


def remove_common_ancestors(ancestor1, ancestor2):
    if not ancestor1 or not ancestor2:
        return ancestor1, ancestor2
    if ancestor1[0] == ancestor2[0]:
        return remove_common_ancestors(ancestor1[1:], ancestor2[1:])
    else:
        return ancestor1, ancestor2


@dataclass
class Stat:
    min: int
    max: int
    value: int
    natural_increase: int
    natural_increase_step: int

    def update(self, i):
        self.value += i
        self.value = min(self.value, self.max)
        self.value = max(self.value, self.min)

    def discretize(self):
        if self.value < self.max // 4:
            return 0
        elif self.value < self.max // 2:
            return 1
        elif self.value < 3 * self.max // 4:
            return 2
        else:
            return 3


class Stats:
    def __init__(self):
        # TODO: This is hard-coded, we should change this in the
        self._energy = Stat(0, 20, 20, -1, 2)
        self._health = Stat(0, 20, 20, 1, 5)
        self._joy = Stat(0, 20, 20, 1, 10)
        self._anger = Stat(0, 10, 0, -1, 3)
        self._fear = Stat(0, 10, 0, -1, 3)
        self._sadness = Stat(0, 10, 0, -1, 3)
        self._stats = [self._energy, self._health, self._joy, self._anger, self._fear, self._sadness]

    @lru_cache(maxsize=None)
    def vector_to_id(self, vector):
        id = 0
        for i in range(4):
            id += vector[i] * 4 ** i
        return id

    @lru_cache(maxsize=None)
    def id_to_vector(self, id):
        vector = []
        for i in range(4):
            vector.append(id % 4)
            id = id // 4
        return np.array(vector[::-1])

    @lru_cache(maxsize=None)
    def compute_distance(self, i, j):
        return np.linalg.norm(self.id_to_vector(i) - self.id_to_vector(j))

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


def flatten(lst):
    return list(chain.from_iterable(lst))


class SymbolicEnv(gym.Env):
    _DISTANCE_QUERY = """
    SELECT ?distanceValue
    WHERE {
        ?distance ??1 ??2 .
        ?distance ??1 ??3 .
        ?distance ??4 ?distanceValue .
    }
    """

    _ALL_ANCESTORS = """
    SELECT DISTINCT ?ancestor
    WHERE {
        ?? rdf:type/rdfs:subClassOf* ?ancestor .
        ?ancestor a owl:Class .
    }
    """

    _MAX_DISTANCE_VALUE = """
        SELECT (MAX(?distanceValue) AS ?maxDistanceValue)
        WHERE {
            ?thing a/rdfs:subClassOf* ??1 .
            ?distance ??2 ?thing .
            ?distance ??3 ?distanceValue .
        }
    """

    _GET_CONSEQUENCE = """
            SELECT ?consequence
            WHERE {
                ?consequence a/rdfs:subClassOf* ??1 .
                ?consequence ??4 ??2 .
                ?consequence ??5 ??3 .
            }
        """

    def __init__(self, ontology_file: str = "ontology.yaml", individuals_file: Optional[str] = "individuals.yaml"):
        self.onto: Ontology = ypo.OntologyManager(ontology_file).onto
        if individuals_file:
            self._load_individuals(individuals_file)
        self._create_distances_relations()
        self._create_effect_relations()
        self._create_consequence_relations()

        sync_reasoner()

        self.individuals = self.onto.Entity.instances()
        self.actions = self.onto.Action.instances()

        for i, _individual in enumerate(self.individuals):
            _individual.id = i
            _individual.ancestors = self._ancestors(_individual)

        self.distances = self._compute_distances()
        self.current_thing: Optional[EntityClass] = None
        self.stats = Stats()
        self.observation_space = spaces.MultiDiscrete([len(self.stats), len(self.individuals)])
        self.action_space = spaces.Discrete(len(self.actions))

        self.step_counter = 0

    def _load_individuals(self, file="individuals.yaml"):
        with open(file, "r") as stream:
            data = yaml.safe_load(stream)
        for k, v in data.items():
            entity = self.onto[v["type"]](k) if "type" in v else self.onto[k]
            if "attrs" in v:
                for property, property_value in v["attrs"].items():
                    if getattr(entity, "INDIRECT_" + property, None):
                        raise RuntimeError(f"The property {property} is already defined for {k}")
                    setattr(entity, property, self.onto[property_value])
            if "consequences" in v:
                for actions, effects in v["consequences"].items():
                    for action in actions.split(","):
                        action = self.onto[action]
                        consequence = self.onto.Consequence(f"consequence({action.name},{entity.name})")
                        for effect in effects:
                            effect = self.onto.Effect(effect)
                            consequence.hasConsequenceEffect.append(effect)

    def _create_distances_relations(self):
        pattern = r"\((.*?)\)"
        for distance in self.onto.Distance.instances():
            match = re.search(pattern, distance.name)
            if match:
                args = match.group(1).split(',')

                distance.hasThing = [self.onto[arg] for arg in args[:-1]]
                distance.hasDistanceValue = float(args[-1])

    def _create_effect_relations(self):
        pattern = r"\((.*?)\)"
        for effect in self.onto.Effect.instances():
            match = re.search(pattern, effect.name)
            if match:
                args = match.group(1).split(',')
                effect.gives = self.onto[args[0]]
                effect.hasEffectValue = int(args[1])

    def _create_consequence_relations(self):
        pattern = r"\((.*?)\)"
        for consequence in self.onto.Consequence.instances():
            match = re.search(pattern, consequence.name)
            if match:
                args = match.group(1).split(',')
                consequence.hasConsequenceAction = self.onto[args[0]]
                consequence.hasConsequenceEntity = self.onto[args[1]]

    def _distance(self, x: EntityClass, y: EntityClass):
        if x == y:
            return 0
        result = \
            list(default_world.sparql(self._DISTANCE_QUERY, [self.onto.hasThing, x, y, self.onto.hasDistanceValue]))[0]
        return result[0] if result else 1

    def _max_distance(self, _class: ThingClass):
        result = \
            list(
                default_world.sparql(self._MAX_DISTANCE_VALUE, [_class, self.onto.hasThing, self.onto.hasDistanceValue])
            )[0]
        return result[0] if result else 1

    def _get_consequence(self, _action, _entity):
        result = list(
            default_world.sparql(self._GET_CONSEQUENCE, [
                self.onto.Consequence,
                _action,
                _entity,
                self.onto.hasConsequenceAction,
                self.onto.hasConsequenceEntity
            ])
        )
        result = flatten(result)
        return result[0] if result else None

    def _ancestors(self, x: EntityClass):
        return flatten(default_world.sparql(self._ALL_ANCESTORS, [x]))

    def _compute_distances(self):
        distances = np.zeros((len(self.individuals), len(self.individuals)))

        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                tree_distance = sum(len(ancestors) for ancestors in remove_common_ancestors(
                    self.individuals[i].ancestors,
                    self.individuals[j].ancestors)
                                    )
                properties_i = set([p.name for p in list(self.individuals[i].INDIRECT_get_properties())])
                properties_j = set([p.name for p in list(self.individuals[i].INDIRECT_get_properties())])
                properties_intersect = properties_i.intersection(properties_j)
                properties_symmetric_difference = properties_i.symmetric_difference(properties_j)
                properties_distance = 0
                for p in properties_intersect:
                    attr = "INDIRECT_" + p
                    attr_i = getattr(self.individuals[i], attr)
                    attr_j = getattr(self.individuals[j], attr)
                    if attr_i is None:
                        raise RuntimeError(f"Property {p} is not defined for {self.individuals[i]}")
                    if attr_j is None:
                        raise RuntimeError(f"Property {p} is not defined for {self.individuals[j]}")
                    properties_distance += \
                        self._distance(attr_i, attr_j) / \
                        self._max_distance(getattr(self.individuals[i], attr).is_a[0])
                properties_distance += len(properties_symmetric_difference)

                distances[i, j] = tree_distance + properties_distance
                distances[j, i] = tree_distance + properties_distance

        #distances = distanes**2
        distances = distances / distances.max()

        return distances

    @lru_cache(maxsize=None)
    def dist(self, observation1, observation2):
        return (self.stats.compute_distance(observation1[0], observation2[0])) + \
               self.distances[observation1[1], observation2[1]]

    def save(self, file_name: str = "ontology.owl"):
        self.onto.save(file_name)

    def render(self):
        pass

    def step(self, action):
        action = self.actions[action]
        consequence = self._get_consequence(action, self.current_thing)
        effects = consequence.hasConsequenceEffect if consequence else [self.onto["effect(sadness,1)"]]
        for effect in effects:
            self.stats.update(effect)
        self.current_thing = random.choice(self.individuals)
        self.stats.natural_increase_stats(self.step_counter)
        self.step_counter += 1
        return self.get_obs(), self.stats.get_reward(), self.stats.is_terminated(), {}

    def get_obs(self):
        return np.array([self.stats.get_obs(), self.individuals.index(self.current_thing)])

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ):
        random.seed(seed)
        self.stats.reset(seed)
        self.step_counter = 0
        self.current_thing = random.choice(self.individuals)
        return self.get_obs()


def plot_distance_matrix(matrix, labels):
    # Create the figure and axes
    fig, ax = plt.subplots()

    # Create the heatmap using the matrix and custom colormap
    im = ax.imshow(matrix, cmap='Reds')

    # Add labels to the rows and columns
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the x-axis labels
    plt.xticks(rotation=90)

    # Add a colorbar
    fig.colorbar(im)

    # Show the plot
    plt.show()


def plot_rewards(rewards_list):
    x = np.linspace(0, len(rewards_list), len(rewards_list))
    plt.figure(figsize=(12, 8))
    plt.plot(x, rewards_list)
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.title('Evolution of reward through time')
    plt.show()


def plot_statistics(_stats_list):
    # TODO: This is hard-coded, we should change this
    stats_names = ['energy', 'health', 'joy', 'anger', 'fear', 'sadness']
    new_stats_lists = np.zeros((len(_stats_list[0]), len(_stats_list)))
    for i in range(len(_stats_list)):
        for j in range(len(stats_list[0])):
            new_stats_lists[j][i] = _stats_list[i][j]
    x = np.linspace(0, new_stats_lists.shape[1], new_stats_lists.shape[1])
    plt.figure(figsize=(12, 8))
    # A verifier
    for idx, stat in enumerate(new_stats_lists):
        plt.plot(x, stat, label=f'{stats_names[idx]}')
    plt.xlabel('Iterations')
    plt.ylabel('Stats values')
    plt.title('Evolution of statistics through time')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    env = SymbolicEnv()
    print(len(env.individuals))
    #plot_distance_matrix(env.distances, [i.name for i in env.individuals])
    check_env(env)

    observation = env.reset()

    for i in range(1000):
        observation, reward, terminated, info = env.step(env.action_space.sample())
        print(i, [s.value for s in env.stats._stats], reward)

        if terminated:
            break
    env.close()
