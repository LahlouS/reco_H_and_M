from implicit.als import AlternatingLeastSquares
from scipy.sparse import load_npz
import numpy as np
import pandas as pd
import json

def load_interaction_matrix(filepath):
    interaction_matrix = load_npz(filepath)
    return interaction_matrix


