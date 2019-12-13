import json


def load_params(fname):
  with open(fname, 'r') as f:
    params = json.load(f)
    return params
