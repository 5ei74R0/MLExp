import random

import numpy
import torch

SEED = 3407


def fix_random_seed(specified_seed: int = SEED):
    random.seed(specified_seed)
    numpy.random.seed(seed=specified_seed)
    torch.manual_seed(seed=specified_seed)
    torch.cuda.manual_seed_all(seed=specified_seed)
