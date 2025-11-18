#Maher Tantouch (Grouphe BMC-BIM)

import numpy as np
import matplotlib.pyplot as plt

import utils

def discretise(x, n_etats):
    intervalle = 360 / n_etats
    return np.floor(np.array(x) / intervalle).astype(int).tolist()


