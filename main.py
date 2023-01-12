from itertools import chain
from pprint import pprint as print
import re
import numpy as np
import yamlpyowl as ypo
from owlready2 import *
import random
from Agent import Agent


def flatten(lst):
    return list(chain.from_iterable(lst))


ALL_ANCESTORS = """
SELECT DISTINCT ?ancestor
WHERE {
    ?? rdf:type/rdfs:subClassOf* ?ancestor .
    ?ancestor a owl:Class .
}
"""

def encounter(agent, instance):
    print("Encountering :")
    print(instance.name)
    chosen_action = agent.get_random_action()
    print("Chosen action :")
    print(chosen_action.name)
    #Requete Sparql pour récupérer la conséquence associée
    #Update l'état de l'agent selon les effets de la conséquence

if __name__ == "__main__":
    onto: Ontology = ypo.OntologyManager("ontology.yaml").onto
    print(list(onto.Organism.instances()))
    print(list(onto.properties()))

    pattern = r"\((.*?)\)"

    for distance in onto.Distance.instances():
        match = re.search(pattern, distance.name)
        if match:
            args = match.group(1).split(',')
            distance.hasThing = [onto[arg] for arg in args]

    sync_reasoner()

    _ALL_ANCESTORS = """
SELECT ?distanceValue
WHERE {
    ?distance ??1 ??2 .
    ?distance ??1 ??3 .
    ?distance ??4 ?distanceValue .
}
    """

    query = default_world.sparql(_ALL_ANCESTORS, [onto.hasThing, onto.warm, onto.cold, onto.hasDistanceValue])

    #print(list(query))

    def remove_common_ancestors(ancestor1, ancestor2):
        if not ancestor1 or not ancestor2:
            return ancestor1, ancestor2
        if ancestor1[0] == ancestor2[0]:
            return remove_common_ancestors(ancestor1[1:], ancestor2[1:])
        else:
            return ancestor1, ancestor2

    individuals = onto.Entity.instances()

    for i, individual in enumerate(individuals):
        individual.id = i
        individual.ancestors = flatten(default_world.sparql(ALL_ANCESTORS, [individual]))

    distance_matrix = np.zeros((len(individuals), len(individuals)))

    for i in range(len(individuals)):
        for j in range(i + 1, len(individuals)):
            tree_distance = sum(len(ancestors) for ancestors in remove_common_ancestors(
                individuals[i].ancestors,
                individuals[j].ancestors)
            )
            distance_matrix[i, j] = tree_distance
            distance_matrix[j, i] = tree_distance

    #print(distance_matrix)
    agent = Agent(onto, 0.1, 0.9, 1)

    instances = list(onto.Entity.instances())
    max_iter = 10
    i = 0

    while (i < max_iter):
        chosen_instance = random.choice(instances)
        encounter(agent, chosen_instance)
        i = i+1

    onto.save("ontology.owl")
