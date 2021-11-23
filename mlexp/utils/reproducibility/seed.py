import random

import numpy
import torch

SEED = 3407


def fix_random_seeds(specified_seed: int = SEED):
    """fix random seed of torch, numpy, python

    Args:
        specified_seed (int, optional): random seed. Defaults to SEED=3407.
    """
    random.seed(specified_seed)
    numpy.random.seed(seed=specified_seed)
    torch.manual_seed(seed=specified_seed)
    torch.cuda.manual_seed_all(seed=specified_seed)
