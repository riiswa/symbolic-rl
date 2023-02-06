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

from stats import Stats


def remove_common_ancestors(ancestor1, ancestor2):
    """
    Remove common ancestors from two lists.

    Given two lists `ancestor1` and `ancestor2`,
    this function removes any common elements at the beginning of both lists.

    Args:
    ancestor1 (list): The first list of ancestors
    ancestor2 (list): The second list of ancestors

    Returns:
    tuple: A tuple containing two lists of ancestors with common elements removed.

    Example:
    >>> remove_common_ancestors([1, 2, 3], [1, 2, 4])
    ([3], [4])
    >>> remove_common_ancestors([1, 2, 3], [1, 2])
    ([3], [])
    """
    if not ancestor1 or not ancestor2:
        return ancestor1, ancestor2
    if ancestor1[0] == ancestor2[0]:
        return remove_common_ancestors(ancestor1[1:], ancestor2[1:])
    else:
        return ancestor1, ancestor2


def flatten(lst):
    """
    Flatten a list of lists into a single list.

    Given a list of lists `lst`, this function returns a single list with all elements from the sublists.

    Args:
    lst (list): The list of lists to be flattened.

    Returns:
    list: A single list with all elements from the sublists.

    Example:
    >>> flatten([[1, 2], [3, 4], [5, 6]])
    [1, 2, 3, 4, 5, 6]
    >>> flatten([[1], [2, 3], []])
    [1, 2, 3]
    """
    return list(chain.from_iterable(lst))


class SymbolicEnv(gym.Env):
    """
    A symbolic environment for reinforcement learning.

    This class is a custom implementation of OpenAI Gym's `gym.Env` class. It creates an environment for reinforcement
    learning problems that require symbolic reasoning. The class can be extended to implement specific environments by
    defining the methods `reset`, `step`, and `render`.
    """

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
        """
            Load individuals from a YAML file and add them to the environment's ontology.

            Given a YAML file `file`, this function loads individuals and their attributes and consequences.
            The function adds the loaded individuals to the environment's ontology.

            If an individual has the "type" property, it creates an instance of the specified type.
            If an individual has the "attrs" property, it adds the attributes to the individual.
            If an individual has the "consequences" property, it creates consequences for the individual and adds the
            effects to the consequences.

            Args:
            file (str): The name of the YAML file to be loaded (default is "individuals.yaml").

            Raises:
            RuntimeError: If an attempt is made to add a property that is already defined for an individual.
            """
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
        """
        Create relationships between instances of Distance and Thing.

        This function creates relationships between instances of the `Distance` and `Thing` classes.
        The function uses a regular expression pattern to extract the names of the things involved in the distance
        relationship and the value of the distance from the name of the Distance instance.

        The extracted names are used to create a list of Thing instances, which are then set as the `hasThing`
        attribute of the Distance instance. The extracted distance value is set as the `hasDistanceValue` attribute of
        the Distance instance.
        """
        pattern = r"\((.*?)\)"
        for distance in self.onto.Distance.instances():
            match = re.search(pattern, distance.name)
            if match:
                args = match.group(1).split(',')

                distance.hasThing = [self.onto[arg] for arg in args[:-1]]
                distance.hasDistanceValue = float(args[-1])

    def _create_effect_relations(self):
        """
            Create relationships between instances of Effect and Thing.

            This function creates relationships between instances of the `Effect` and `Thing` classes. The function uses
            a regular expression pattern to extract the names of the things involved in the effect relationship and the
            value of the effect from the name of the Effect instance.

            The extracted name is used to create a Thing instance, which is then set as the `gives` attribute of the
            Effect instance. The extracted effect value is set as the `hasEffectValue` attribute of the Effect instance.
            """
        pattern = r"\((.*?)\)"
        for effect in self.onto.Effect.instances():
            match = re.search(pattern, effect.name)
            if match:
                args = match.group(1).split(',')
                effect.gives = self.onto[args[0]]
                effect.hasEffectValue = int(args[1])

    def _create_consequence_relations(self):
        """
        Create relationships between instances of Consequence, Action, and Entity.

        This function creates relationships between instances of the `Consequence`, `Action`, and `Entity` classes.
        The function uses a regular expression pattern to extract the names of the actions and entities involved in the
        consequence relationship from the name of the Consequence instance.

        The extracted names are used to create instances of the Action and Entity classes, which are then set as the
        `hasConsequenceAction` and `hasConsequenceEntity` attributes of the Consequence instance, respectively.
        """
        pattern = r"\((.*?)\)"
        for consequence in self.onto.Consequence.instances():
            match = re.search(pattern, consequence.name)
            if match:
                args = match.group(1).split(',')
                consequence.hasConsequenceAction = self.onto[args[0]]
                consequence.hasConsequenceEntity = self.onto[args[1]]

    def _distance(self, x: EntityClass, y: EntityClass):
        """
        Calculate the distance between two Entity instances.

        This function calculates the distance between two `Entity` instances using a SPARQL query.
        If the two instances are the same, the distance is 0.
        Otherwise, the function executes a SPARQL query to retrieve the distance between the instances.
        If the query returns no results, the distance is set to 1.

        Args:
        x (EntityClass): The first Entity instance.
        y (EntityClass): The second Entity instance.

        Returns:
        int: The distance between the two Entity instances.
        """
        if x == y:
            return 0
        result = \
            list(default_world.sparql(self._DISTANCE_QUERY, [self.onto.hasThing, x, y, self.onto.hasDistanceValue]))[0]
        return result[0] if result else 1

    def _max_distance(self, _class: ThingClass):
        """
            Calculate the maximum distance for a class of Things.

            This function calculates the maximum distance for a class of Things using a SPARQL query.
            The function executes a SPARQL query to retrieve the maximum distance for the specified class of Things.
            If the query returns no results, the maximum distance is set to 1.

            Args:
            _class (ThingClass): The class of Things to calculate the maximum distance for.

            Returns:
            int: The maximum distance for the specified class of Things.
            """
        result = \
            list(
                default_world.sparql(self._MAX_DISTANCE_VALUE, [_class, self.onto.hasThing, self.onto.hasDistanceValue])
            )[0]
        return result[0] if result else 1

    def _get_consequence(self, _action, _entity):
        """
        This method returns the consequence of a specific action on a specific entity.

        Args:
        _action (ActionClass): The action to be performed.
        _entity (EntityClass): The entity on which the action will be performed.

        Returns:
        ConsequenceClass: The consequence of the action on the entity. If no consequence exists, returns None.
        """
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
        """
        Given an instance of the EntityClass, return its ancestors in the class hierarchy of the ontology.

        Args:
        x (EntityClass): Instance of the EntityClass to find its ancestors.

        Returns:
        list: List of all the ancestors of the instance x in the class hierarchy of the ontology.
        """
        return flatten(default_world.sparql(self._ALL_ANCESTORS, [x]))

    def _compute_distances(self):
        """
        This method computes the distances between all instances of the individuals in the ontology.
        It uses a combination of tree distance and properties distance to calculate the distances.
        Tree distance is the sum of the length of the ancestors of each individual.
        Properties distance is the sum of the distances between the properties of each individual.
        The distances are normalized by the maximum distance value.

        Returns:
            numpy.ndarray: A 2D array representing the distances between all individuals.
        """
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

        distances = distances / distances.max()

        return distances

    @lru_cache(maxsize=None)
    def dist(self, observation1, observation2):
        """
        Calculate the distance between two observations.

        Parameters
        ----------
        observation1 : tuple
            A tuple of the form (stats, individual) representing the first observation.
        observation2 : tuple
            A tuple of the form (stats, individual) representing the second observation.

        Returns
        -------
        float
            The distance between the two observations, calculated as the average of the
            distances between their stats and their individuals.

        """
        return (self.stats.dist(observation1[0], observation2[0]) + self.distances[
            observation1[1], observation2[1]]) / 2

    def save(self, file_name: str = "ontology.owl"):
        """
        Saves the ontology to a file.

        Args:
         file_name (str, optional): The name of the file to save the ontology to. Defaults to "ontology.owl".
        """
        self.onto.save(file_name)

    def render(self):
        pass  # TODO

    def step(self, action):
        """
        Steps the environment based on the given action.

        Parameters:
        action (int): index of the selected action.

        Returns:
        tuple: a tuple of (observation, reward, done, info)
            observation (tuple): a tuple of observation data.
            reward (float): the reward obtained by taking the action.
            done (bool): whether the episode has terminated.
            info (dict): additional information (not used).
        """
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
        """
        The get_obs method returns the observation for the current state of the environment.

        Returns:
        numpy.ndarray: An array of two elements: the first element is the observation of the current stats and the
        second element is the index of the current individual.
        """
        return np.array([self.stats.get_obs(), self.individuals.index(self.current_thing)])

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ):
        """
        Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed for environment randomness, by default None
        options : dict[str, Any], optional
            Dictionary of options to reset the environment with, by default None

        Returns
        -------
        np.array
            Observation of the environment
        """
        random.seed(seed)
        self.stats.reset(seed)
        self.step_counter = 0
        self.current_thing = random.choice(self.individuals)
        return self.get_obs()


def plot_distance_matrix(matrix, labels):
    """
    Plot the distance matrix using a heatmap.

    Parameters:
    - matrix (numpy array): The distance matrix to be plotted.
    - labels (list): A list of strings to use as labels for the rows and columns of the matrix.
    """
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
    env.save()
    print(len(env.individuals))
    check_env(env)
    observation = env.reset()

    for i in range(1000):
        observation, reward, terminated, info = env.step(env.action_space.sample())
        print(i, [s.value for s in env.stats._stats], reward)

        if terminated:
            break
    env.close()
