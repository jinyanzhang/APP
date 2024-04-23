import json
import os
import sys
import random
import pickle
import shutil

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from typing import Any, IO


def copy_files(source_dir, dest_dir, extensions=['.py', '.sh']):
    for filename in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, filename)):
            _, ext = os.path.splitext(filename)
            if ext in extensions:
                shutil.copy2(os.path.join(source_dir, filename), dest_dir)


def print_args(args):
    print("[INFO] Input arguments:")
    for key, val in args.items():
        print(f"[INFO]   {key}: {val}")
        

def set_random_seed(seed):
    """Sets random seed for training reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a+")
 
    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()
    
    def flush(self):
        pass

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def count_param_numbers(model):
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content