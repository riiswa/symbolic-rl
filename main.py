from owlready2 import *
import yamlpyowl as ypo
from pprint import pprint as print

if __name__ == "__main__":
    onto: Ontology = ypo.OntologyManager("ontology.yaml").onto
    print(list(onto.Organism.instances()))
    print(list(onto.properties()))
    print(onto.almond.INDIRECT_has_wetness)
    sync_reasoner()
    print(onto.almond)

    onto.save("ontology.owl")
