import random
from dataclasses import dataclass
from typing import SupportsFloat, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

import yamlpyowl as ypo
from gymnasium.core import ActType, ObsType, RenderFrame
from owlready2 import *


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

    def update(self, i):
        self.value += i
        self.value = min(self.value, self.max)
        self.value = max(self.value, self.min)


class Stats:
    def __init__(self):
        # TODO: This is hard-coded, we should change this in the
        self._energy = Stat(0, 100, 100)
        self._health = Stat(0, 100, 100)
        self._joy = Stat(0, 100, 100)
        self._anger = Stat(0, 50, 0)
        self._fear = Stat(0, 50, 0)
        self._sadness = Stat(0, 50, 0)
        self.observation_matrix = [(i, j, k) for k in range(2) for j in range(2) for i in range(2)]

    def get_space(self):
        return len(self.observation_matrix)

    def energy(self):
        return self._energy.value

    def health(self):
        return self._health.value

    def mood(self):
        return min(self._joy.value - self._anger.value - self._fear.value - self._sadness.value, 0)

    def update(self, effect):
        getattr(self, "_" + effect.gives.name).update(effect.hasEffectValue)

    def get_obs(self):
        def f(b: bool) -> int:
            return 1 if b else 0

        return self.observation_matrix.index((f(self.energy() > 25), f(self.health() > 25), f(self.mood() > 25)))

    def reset(self):
        self._energy.value = 100
        self._health.value = 100
        self._joy.value = 100
        self._fear.value = 0
        self._anger.value = 0
        self._sadness.value = 0


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

    def __init__(self, ontology_file: str = "ontology.yaml"):
        self.onto: Ontology = ypo.OntologyManager(ontology_file).onto
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
        self.observation_space = spaces.MultiDiscrete([self.stats.get_space(), len(self.individuals)])
        self.action_space = spaces.Discrete(len(self.actions))

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

    def _get_consequence(self, action, _entity):
        result = \
            list(
                default_world.sparql(self._GET_CONSEQUENCE, [
                    self.onto.Consequence,
                    action,
                    _entity,
                    self.onto.hasConsequenceAction,
                    self.onto.hasConsequenceEntity
                ])
            )[0]

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
                    properties_distance += \
                        self._distance(getattr(self.individuals[i], attr), getattr(self.individuals[j], attr)) / \
                        self._max_distance(getattr(self.individuals[i], attr).is_a[0])
                properties_distance += len(properties_symmetric_difference)

                distances[i, j] = tree_distance + properties_distance
                distances[j, i] = tree_distance + properties_distance

        return distances

    def save(self, file_name: str = "ontology.owl"):
        self.onto.save(file_name)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = self.actions[action]
        self.current_thing

    def _get_obs(self):
        return np.array([self.stats.get_obs(), self.individuals.index(self.current_thing)])

    def reset(self, seed=None):
        random.seed(seed)
        self.stats.reset()
        self.current_thing = random.choice(self.individuals)


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


if __name__ == "__main__":
    env = SymbolicEnv()
    print(env.individuals)
    print(env.distances)
    print(env._get_consequence(env.onto.eat, env.onto.medicinalHerb).hasConsequenceEffect)
    env.save()

    # plot_distance_matrix(env.distances, env.individuals)
