import pickle
import numpy as np


with open('arrays.pkl', 'rb') as f:
    obs, lstm0, lstm1 = pickle.load(f)
print(obs)

print(lstm0)
print(lstm1)
