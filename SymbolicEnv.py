import gymnasium as gym
import numpy as np
from gymnasium import spaces

import yamlpyowl as ypo
from owlready2 import *


def remove_common_ancestors(ancestor1, ancestor2):
    if not ancestor1 or not ancestor2:
        return ancestor1, ancestor2
    if ancestor1[0] == ancestor2[0]:
        return remove_common_ancestors(ancestor1[1:], ancestor2[1:])
    else:
        return ancestor1, ancestor2


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

    def __init__(self, ontology_file: str = "ontology.yaml"):
        self.onto: Ontology = ypo.OntologyManager(ontology_file).onto
        self._create_distances_relations()

        sync_reasoner()

        self.individuals = self.onto.Entity.instances()

        for i, _individual in enumerate(self.individuals):
            _individual.id = i
            _individual.ancestors = self._ancestors(_individual)

        self.distances = self._compute_distances()

    def _create_distances_relations(self):
        pattern = r"\((.*?)\)"
        for distance in self.onto.Distance.instances():
            match = re.search(pattern, distance.name)
            if match:
                args = match.group(1).split(',')
                distance.hasThing = [self.onto[arg] for arg in args]

    def _distance(self, x: EntityClass, y: EntityClass):
        if x == y:
            return 0
        result = default_world.sparql(self._DISTANCE_QUERY, [self.onto.hasThing, x, y, self.onto.hasDistanceValue])[0]
        return result[0] if result else 1

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
                distances[i, j] = tree_distance
                distances[j, i] = tree_distance

        return distances

    def save(self, file_name: str = "ontology.owl"):
        self.onto.save(file_name)


if __name__ == "__main__":
    env = SymbolicEnv()
    print(env.distances)


