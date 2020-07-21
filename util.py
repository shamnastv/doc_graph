import yaml


def read_param(filename):
    with open(filename, 'r') as fd:
        data = yaml.safe_load(fd)
    return data
