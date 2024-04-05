

######### เอาไว้ดู data เฉยๆ ##########
import pickle
import numpy as np

with open('all_data.pickle', 'rb') as file:
    data = pickle.load(file)

print(data)
