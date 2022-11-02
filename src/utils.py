import yaml


def merge_yaml_data(filepaths):
    data = {}
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            content = yaml.load(f, yaml.Loader)
        data |= content
    return data
