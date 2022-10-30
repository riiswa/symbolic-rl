import yaml


def load_ontology(filepaths):
    ontology = {}
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            content = yaml.load(f, yaml.BaseLoader)
        ontology |= content
    return ontology
